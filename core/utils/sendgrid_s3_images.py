import os
import boto3
import uuid
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
import logging
from PIL import Image
import io
from django.core.files.uploadedfile import InMemoryUploadedFile

logger = logging.getLogger(__name__)

# Helper to get file size for both InMemoryUploadedFile and BytesIO
def get_file_size(file):
    if hasattr(file, 'size'):
        return file.size
    elif hasattr(file, 'getvalue'):
        return len(file.getvalue())
    else:
        return 0

class S3ImageService:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')
        self.folder_prefix = 'email-images/'
        
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is required")
    
    def validate_image(self, image_file):
        """Validate uploaded image file"""
        # Check file size (max 5MB)
        if get_file_size(image_file) > 5 * 1024 * 1024:
            raise ValueError("Image file size must be less than 5MB")
        
        # Check file type
        allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
        if image_file.content_type not in allowed_types:
            raise ValueError(f"Unsupported image type: {image_file.content_type}")
        
        # Validate image can be opened
        try:
            image = Image.open(image_file)
            image.verify()
            image_file.seek(0)  # Reset file pointer after verification
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
    
    def optimize_image(self, image_file, max_width=800, quality=85):
        """Optimize image for email use"""
        try:
            # Reset file pointer to beginning
            image_file.seek(0)
            image = Image.open(image_file)
            
            # Convert RGBA to RGB if necessary (for JPEG)
            if image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = rgb_image
            
            # Resize if too wide
            if image.width > max_width:
                ratio = max_width / image.width
                new_height = int(image.height * ratio)
                image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to memory
            output = io.BytesIO()
            format_map = {
                'image/jpeg': 'JPEG',
                'image/png': 'PNG',
                'image/gif': 'GIF',
                'image/webp': 'WEBP'
            }
            
            image_format = format_map.get(image_file.content_type, 'JPEG')
            
            if image_format == 'JPEG':
                image.save(output, format=image_format, quality=quality, optimize=True)
            else:
                image.save(output, format=image_format, optimize=True)
            
            # Reset to beginning so it can be read
            output.seek(0)
            return output, image_format.lower()
            
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            # If optimization fails, return original with proper file pointer reset
            image_file.seek(0)
            return image_file, image_file.content_type.split('/')[-1]
    
    def upload_image(self, image_file, campaign_id=None, step_id=None):
        """Upload image to S3 and return public URL"""
        try:
            # Validate image
            self.validate_image(image_file)
            
            # Optimize image
            optimized_image, file_extension = self.optimize_image(image_file)
            
            # Store the optimized image size BEFORE upload (while file is still accessible)
            optimized_size = get_file_size(optimized_image)
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            
            # Build file path
            path_parts = [self.folder_prefix]
            if campaign_id:
                path_parts.append(f"campaign_{campaign_id}/")
            if step_id:
                path_parts.append(f"step_{step_id}/")
            
            filename = f"{timestamp}_{unique_id}.{file_extension}"
            s3_key = "".join(path_parts) + filename
            
            # Upload to S3
            extra_args = {
                'ContentType': image_file.content_type,
                'CacheControl': 'max-age=31536000',  # Cache for 1 year
                'Metadata': {
                    'uploaded_at': datetime.now().isoformat(),
                    'original_filename': getattr(image_file, 'name', 'unknown'),
                    'campaign_id': str(campaign_id) if campaign_id else '',
                    'step_id': str(step_id) if step_id else '',
                }
            }
            
            self.s3_client.upload_fileobj(
                optimized_image,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            # Generate public URL
            public_url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
            
            logger.info(f"Successfully uploaded image to S3: {s3_key}")
            
            return {
                'url': public_url,
                's3_key': s3_key,
                'filename': filename,
                'size': optimized_size,  # Use the size we captured earlier
                'content_type': image_file.content_type
            }
            
        except ClientError as e:
            logger.error(f"AWS S3 error uploading image: {e}")
            raise Exception(f"Failed to upload image to S3: {e}")
        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            raise Exception(f"Failed to upload image: {e}")
    
    def delete_image(self, s3_key):
        """Delete image from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Successfully deleted image from S3: {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting image from S3: {e}")
            return False
    
    def get_presigned_url(self, s3_key, expiration=3600):
        """Generate presigned URL for private images (if needed)"""
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return response
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

# Utility function to embed images in HTML
def embed_images_in_html(html_content, image_urls):
    """
    Replace image placeholders in HTML with actual S3 URLs
    Expected placeholders: {{image_1}}, {{image_2}}, etc.
    """
    if not image_urls:
        return html_content
    
    for i, url in enumerate(image_urls, 1):
        placeholder = f"{{{{image_{i}}}}}"
        html_content = html_content.replace(placeholder, url)
    
    return html_content

def auto_embed_images_in_html(html_content, image_data_list):
    """
    Automatically embed images in HTML content
    Looks for <img> tags with src="placeholder" or adds images at the end
    """
    if not image_data_list:
        return html_content
    
    # Simple approach: replace placeholder img tags or append images
    for i, image_data in enumerate(image_data_list):
        img_tag = f'<img src="{image_data["url"]}" alt="Email Image {i+1}" style="max-width: 100%; height: auto; display: block; margin: 10px 0;">'
        
        # Try to replace placeholder
        placeholder_patterns = [
            f'<img src="placeholder_{i+1}"',
            f'<img src="{{{{image_{i+1}}}}}',
            '<img src="placeholder"'
        ]
        
        replaced = False
        for pattern in placeholder_patterns:
            if pattern in html_content:
                # Replace the entire img tag
                import re
                html_content = re.sub(
                    f'{pattern}[^>]*>',
                    img_tag,
                    html_content,
                    count=1
                )
                replaced = True
                break
        
        # If no placeholder found, append at the end of body
        if not replaced:
            if '</body>' in html_content:
                html_content = html_content.replace('</body>', f'{img_tag}</body>')
            else:
                html_content += img_tag
    
    return html_content