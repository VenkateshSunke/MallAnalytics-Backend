import os
import logging
import boto3
import cv2
import numpy as np
from botocore.exceptions import ClientError
from ultralytics import YOLO
from datetime import datetime
from retinaface import RetinaFace


logger = logging.getLogger(__name__)



class AWSRekognitionService:
    """AWS Rekognition service for face detection"""
    
    def __init__(self):
        self.rekognition_client = None
        self.aws_enabled = False
        self.api_calls_count = 0
        self.frame_api_calls_count = 0
        self.last_api_call_time = 0.0
        self.face_detection_interval = 2.0
        self.last_face_detection_time = 0.0
        self.is_exporting = False
        self.last_face_detections = []
        self.collection_id = os.getenv('AWS_REKOGNITION_COLLECTION_ID', 'my-face-collection')
        self.debug_mode = False

        # Initialize RetinaFace (no need to load model explicitly)
        self.retinaface_loaded = True
        
        # # Initialize YOLOv8 face model for better face detection
        # self.yolo_face_model = None
        # self.load_yolo_face_model()
    
    def load_yolo_face_model(self):
        """Load YOLOv8 face detection model"""
        try:
            # Try to load a face-specific model first, fall back to general model
            try:
                self.yolo_face_model = YOLO('yolov8n-face.pt')  # Face-specific model
                logger.info("Loaded YOLOv8 face-specific model")
            except:
                self.yolo_face_model = YOLO('yolov8n.pt')  # General model
                logger.info("Loaded YOLOv8 general model (will filter for person class)")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            self.yolo_face_model = None
    
    def enable_aws_rekognition(self, aws_region=None):
        """Enable AWS Rekognition using explicit credentials"""
        try:
            # Get AWS credentials and region from environment variables
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = aws_region or os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'ap-south-1'
            
            logger.info(f"AWS Environment Variables - Access Key: {aws_access_key[:10] if aws_access_key else 'None'}..., Secret Key: {aws_secret_key[:10] if aws_secret_key else 'None'}..., Region: {aws_region}")
            
            if not aws_access_key or not aws_secret_key:
                logger.error("AWS credentials not found in environment variables")
                return False, "AWS credentials not found in environment variables"
            
            logger.info(f"Initializing AWS Rekognition with region: {aws_region}")
            
            # Create Rekognition client with explicit credentials
            self.rekognition_client = boto3.client(
                'rekognition',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            
            # Test the connection
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            _, test_bytes = cv2.imencode('.jpg', test_image)
            
            # self.rekognition_client.detect_faces(
            #     Image={'Bytes': test_bytes.tobytes()},
            #     Attributes=['ALL']
            # )
            
            self.aws_enabled = True
            self.api_calls_count = 0
            self.frame_api_calls_count = 0
            self.last_api_call_time = None
            logger.info("AWS Rekognition enabled - Face detection API calls will be made every 2 seconds")
            return True, "AWS Rekognition enabled successfully"
            
        except ClientError as e:
            self.aws_enabled = False
            self.rekognition_client = None
            return False, f"AWS Rekognition error: {str(e)}"
        except Exception as e:
            self.aws_enabled = False
            self.rekognition_client = None
            return False, f"Error enabling AWS Rekognition: {str(e)}"
    
    def set_export_mode(self, enabled):
        """Enable/disable export mode to control API calls"""
        self.is_exporting = enabled
        if enabled:
            logger.info("AWS Rekognition enabled for face detection in export mode")
        else:
            logger.info("AWS Rekognition disabled for preview mode")
    
    def detect_faces(self, frame, current_time):
        """Detect registered users using YOLOv8 + AWS Rekognition - enhanced for CCTV"""
        if not self.aws_enabled or not self.is_exporting:
            if not self.aws_enabled:
                logger.debug("AWS call prevented: AWS not enabled")
            elif not self.is_exporting:
                logger.debug("AWS call prevented: Not in export mode (preview mode active)")
            return []
        
        # if not self.yolo_face_model:
        #     logger.warning("YOLOv8 model not loaded")
        #     return []
        if not self.retinaface_loaded:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] RetinaFace not loaded")
            return []
        
        try:
            logger.info("Starting enhanced face detection")
            
            # Upscale frame for better face detection
            height, width = frame.shape[:2]
            if width < 1920 or height < 1080:
                scale_factor = max(1920 / width, 1080 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Step 1: Detect all faces in the image using YOLOv8
            # results = self.yolo_face_model(frame, verbose=False)
            detections = RetinaFace.detect_faces(frame)
            
            # Update API call tracking
            self.api_calls_count += 1
            self.frame_api_calls_count += 1
            self.last_api_call_time = current_time
            self.last_face_detection_time = current_time
            
            logger.info(f"YOLOv8 Face Detection Call #{self.api_calls_count} at {current_time:.1f}s")
            
            # Extract detected faces from YOLOv8 results
            detected_faces = []
            # for result in results:
            #     boxes = result.boxes
            #     if boxes is not None:
            #         for box in boxes:
            #             conf = float(box.conf[0])
            #             if conf > 0.3:  # Confidence threshold
            #                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
            #                 # Convert to relative coordinates (like AWS format)
            #                 face_bbox = {
            #                     'Left': x1 / frame.shape[1],
            #                     'Top': y1 / frame.shape[0],
            #                     'Width': (x2 - x1) / frame.shape[1],
            #                     'Height': (y2 - y1) / frame.shape[0]
            #                 }
                            
            #                 detected_faces.append({
            #                     'BoundingBox': face_bbox,
            #                     'Confidence': conf * 100  # Convert to percentage
            #                 })
            if isinstance(detections, dict):
                for face_key, face_data in detections.items():
                    # Get face area (bounding box)
                    face_area = face_data['facial_area']
                    x1, y1, x2, y2 = face_area
                    
                    # Get confidence score
                    confidence = face_data.get('score', 0.9)  # RetinaFace doesn't always provide score
                    
                    if confidence > 0.3:  # Confidence threshold
                        # Convert to relative coordinates (like AWS format)
                        face_bbox = {
                            'Left': x1 / frame.shape[1],
                            'Top': y1 / frame.shape[0],
                            'Width': (x2 - x1) / frame.shape[1],
                            'Height': (y2 - y1) / frame.shape[0]
                        }
                        
                        detected_faces.append({
                            'BoundingBox': face_bbox,
                            'Confidence': confidence * 100  # Convert to percentage
                        })
            # Step 2: If faces are found, search for matches in collection
            if detected_faces:
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Processing {len(detected_faces)} detected faces for recognition")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] No faces detected - skipping recognition")
                return []
            
            logger.info(f"Found {len(detected_faces)} faces in frame")
            
            # # Step 2: If faces are found, search for matches in collection
            # if not detected_faces:
            #     logger.info("No faces detected - skipping recognition")
            #     return []

            # Step 2: Process each detected face for recognition
            for i, face_detail in enumerate(detected_faces):
                face_bbox = face_detail['BoundingBox']
                confidence = face_detail['Confidence']
                
                # Skip if confidence is too low (lowered threshold for better detection)
                if confidence < 30.0:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Skipping - confidence too low ({confidence:.1f}%)")
                    continue
                
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Processing face with confidence {confidence:.1f}%")
                
                # Convert relative coordinates to pixel coordinates (ORIGINAL face bbox)
                original_x = int(face_bbox['Left'] * frame.shape[1])
                original_y = int(face_bbox['Top'] * frame.shape[0])
                original_width = int(face_bbox['Width'] * frame.shape[1])
                original_height = int(face_bbox['Height'] * frame.shape[0])
                
                # IMPROVED: Better validation of original face size
                if original_width < 10 or original_height < 10:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Skipping - original face too small ({original_width}x{original_height})")
                    if self.debug_mode:
                        # Save the problematic region for debugging
                        problematic_region = frame[max(0, original_y-10):min(frame.shape[0], original_y+original_height+10), 
                                                 max(0, original_x-10):min(frame.shape[1], original_x+original_width+10)]
                        self.save_debug_face_region(problematic_region, i+1, "too_small_original")
                    continue
                
                # Store original face bbox for IoU calculations (small red box)
                original_face_bbox = (original_x, original_y, original_width, original_height)
                
                # IMPROVED: More aggressive padding for better recognition (50% instead of 30%)
                padding_x = int(original_width * 0.5)
                padding_y = int(original_height * 0.5)
                
                # IMPROVED: Ensure minimum padding for very small faces
                min_padding = 20
                padding_x = max(padding_x, min_padding)
                padding_y = max(padding_y, min_padding)
                
                # Calculate padded region for recognition
                x = max(0, original_x - padding_x)
                y = max(0, original_y - padding_y)
                width = min(frame.shape[1] - x, original_width + 2 * padding_x)
                height = min(frame.shape[0] - y, original_height + 2 * padding_y)
                
                # Ensure coordinates are within frame bounds
                x = max(0, min(frame.shape[1] - width, x))
                y = max(0, min(frame.shape[0] - height, y))
                width = min(width, frame.shape[1] - x)
                height = min(height, frame.shape[0] - y)
                
                # IMPROVED: Better validation of extracted region
                if width <= 0 or height <= 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Skipping - invalid region dimensions ({width}x{height})")
                    continue
                
                # Extract face region for better recognition
                face_region = frame[y:y + height, x:x + width]
                if face_region.size == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Skipping - empty face region")
                    continue
                
                # IMPROVED: More stringent size validation
                min_region_size = 50  # Increased from 20
                if face_region.shape[0] < min_region_size or face_region.shape[1] < min_region_size:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Skipping - face region too small ({face_region.shape[1]}x{face_region.shape[0]})")
                    if self.debug_mode:
                        self.save_debug_face_region(face_region, i+1, "too_small_region")
                    continue
                
                # IMPROVED: Ensure minimum face size and upscale if needed (like face_detection.py)
                min_face_size = 200
                if face_region.shape[0] < min_face_size or face_region.shape[1] < min_face_size:
                    # Calculate scale factor to reach minimum size
                    scale_x = min_face_size / face_region.shape[1]
                    scale_y = min_face_size / face_region.shape[0]
                    scale = max(scale_x, scale_y, 2.0)  # At least 2x upscale
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Upscaling from {face_region.shape[1]}x{face_region.shape[0]} by {scale:.1f}x")
                    face_region = cv2.resize(face_region, None, 
                                           fx=scale, fy=scale, 
                                           interpolation=cv2.INTER_LANCZOS4)
                
                # IMPROVED: Final size validation after upscaling
                if face_region.shape[0] < 100 or face_region.shape[1] < 100:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Skipping - face region still too small after upscaling ({face_region.shape[1]}x{face_region.shape[0]})")
                    continue
                
                # Convert face region to RGB for AWS
                face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                
                # IMPROVED: Validate RGB conversion
                if face_region_rgb.size == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Skipping - RGB conversion failed")
                    continue
                
                # Encode face region with high quality
                try:
                    _, face_encoded = cv2.imencode('.jpg', face_region_rgb, 
                                                 [cv2.IMWRITE_JPEG_QUALITY, 95])
                    face_bytes = face_encoded.tobytes()
                    
                    # IMPROVED: Validate encoded bytes
                    if len(face_bytes) < 1000:  # Minimum reasonable size for a face image
                        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Skipping - encoded image too small ({len(face_bytes)} bytes)")
                        continue
                        
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Face region encoding error: {e}")
                    continue
                
                # IMPROVED: Pre-validate with detect_faces before search (still using AWS for recognition)
                # try:
                #     # Test if the extracted region actually contains a face
                #     test_response = self.rekognition_client.detect_faces(
                #         Image={'Bytes': face_bytes},
                #         Attributes=['ALL']
                #     )
                #     
                #     if not test_response.get('FaceDetails'):
                #         print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Skipping - no face detected in extracted region")
                #         if self.debug_mode:
                #             self.save_debug_face_region(face_region, i+1, "no_face_detected")
                #         continue
                #     
                #     # Update API call tracking for test
                #     self.api_calls_count += 1
                #     
                # except Exception as e:
                #     print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Pre-validation failed: {str(e)}")
                #     continue
                
                # Extract detected faces
                faces = []

                # Step 2: Search for this specific face in the collection
                try:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Searching collection for matches")
                    search_response = self.rekognition_client.search_faces_by_image(
                        CollectionId=self.collection_id,
                        Image={'Bytes': face_bytes},
                        MaxFaces=5,
                        FaceMatchThreshold=50.0
                    )
                    
                    # Update API call tracking
                    self.api_calls_count += 1
                    
                    face_matches = search_response.get('FaceMatches', [])
                    
                    if face_matches:
                        # Step 3: Match found - this is a registered face
                        match = face_matches[0]  # Take the best match
                        face_id = match['Face'].get('FaceId', 'Unknown')
                        external_image_id = match['Face'].get('ExternalImageId', 'Unknown')
                        similarity = match['Similarity']
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: MATCH FOUND!")
                        print(f"  User: {external_image_id}")
                        print(f"  Similarity: {similarity:.1f}%")
                        print(f"  Face ID: {face_id}")
                        print(f"  Original Face Bbox: ({original_x}, {original_y}, {original_width}, {original_height})")
                        print(f"  Padded Region: ({x}, {y}, {width}, {height})")
                        print(f"  Ready for YOLO bounding box logic")
                        
                        # Step 3: Match found - now use YOLO bounding box logic for face-person association
                        # The face detection result will be passed to PersonTracker for IoU calculations
                        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Creating face detection result for YOLO logic")
                        
                        faces.append({
                            'bbox': original_face_bbox,  # Use ORIGINAL face bbox for IoU calculations (small red box)
                            'padded_bbox': (x, y, width, height),  # Padded region for drawing
                            'confidence': similarity,  # Using similarity as confidence
                            'face_id': face_id,
                            'user_id': external_image_id,
                            'detection_type': 'registered_face',
                            'timestamp': current_time
                        })
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Face detection result created - ready for YOLO association")
                    else:
                        # No match found - this is an unknown face
                        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: No match found - unknown face")
                        # Skip unknown faces to avoid cluttering
                
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Face {i+1}: Error searching collection - {str(e)}")
                    continue

            # Step 3: Summary - faces ready for YOLO bounding box logic
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Found {len(faces)} registered faces")
            if faces:
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Ready to pass {len(faces)} faces to YOLO bounding box logic")
            
            self.last_face_detections = faces  # Store for drawing
            return faces
            
            
        except Exception as e:
            logger.error(f"Error in enhanced face detection: {e}")
            return []
    
    def get_api_stats(self):
        """Get API usage statistics"""
        return {
            'calls_count': self.api_calls_count,
            'frame_calls_count': self.frame_api_calls_count,
            'last_call_time': self.last_api_call_time,
            'aws_enabled': self.aws_enabled
        }
    
    def reset_api_counters(self):
        """Reset both API call counters"""
        self.api_calls_count = 0
        self.frame_api_calls_count = 0

    def get_detection_color(self, detection_type, current_time):
        """Get color for drawing based on detection type"""
        if detection_type == 'face':
            return (0, 0, 255)  # Red for face detection
        return (255, 0, 0)  # Blue for other detections
    
    def cleanup_face_detections(self):
        """Clean up face detection state"""
        self.last_face_detections = []
    
    def reset_state(self):
        """Reset AWS service state for clean restart"""
        self.last_face_detections = []
        self.api_calls_count = 0
        self.frame_api_calls_count = 0
        self.last_api_call_time = 0.0
        self.last_face_detection_time = 0.0
        print("AWS service state reset for clean restart")