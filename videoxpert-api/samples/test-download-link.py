"""
VideoXpert Export Downloader

Downloads export files from VideoXpert system using direct URLs.
"""

import requests
import base64
import os
import sys
from urllib.parse import urlparse
import time

# Configuration
VIDEOXPERT_USERNAME = "wise"
VIDEOXPERT_PASSWORD = "W1s3$2025"
EXPORT_URL = "https://190.116.49.5/system/5.2/exports/data/2b6f0870-56c4-4e55-9ef1-64ced9397d88"

def encode_auth_headers(username, password):
    """Encode username and password for VideoXpert authentication headers."""
    user_b64 = base64.b64encode(username.encode('ascii')).decode('ascii')
    pass_b64 = base64.b64encode(password.encode('ascii')).decode('ascii')
    return user_b64, pass_b64

def get_filename_from_url(url):
    """Extract filename from URL or generate a default one."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    if not filename or not filename.endswith('.zip'):
        # Generate filename from export ID
        export_id = parsed.path.split('/')[-1]
        filename = f"export_{export_id}.zip"
    return filename

def download_export(url, username, password, output_filename=None):
    """Download export file from VideoXpert system."""
    
    # Encode authentication headers
    user_b64, pass_b64 = encode_auth_headers(username, password)
    
    # Set up headers
    headers = {
        'X-Serenity-User': user_b64,
        'X-Serenity-Password': pass_b64,
        'User-Agent': 'VideoXpert-Export-Downloader/1.0'
    }
    
    # Determine output filename
    if not output_filename:
        output_filename = get_filename_from_url(url)
    
    print(f"🔐 Authenticating as: {username}")
    print(f"📥 Downloading from: {url}")
    print(f"💾 Output file: {output_filename}")
    print(f"📊 Headers: X-Serenity-User={user_b64[:10]}...")
    print()
    
    try:
        # Disable SSL verification for self-signed certificates
        session = requests.Session()
        session.verify = False
        
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Make request with stream=True for large files
        print("🚀 Starting download...")
        response = session.get(url, headers=headers, stream=True, timeout=30)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Get file size if available
        file_size = response.headers.get('content-length')
        if file_size:
            file_size = int(file_size)
            print(f"📏 File size: {format_file_size(file_size)}")
        
        # Download the file
        downloaded = 0
        chunk_size = 8192  # 8KB chunks
        start_time = time.time()
        
        with open(output_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress
                    if file_size:
                        progress = (downloaded / file_size) * 100
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        print(f"\r📊 Progress: {progress:.1f}% | Downloaded: {format_file_size(downloaded)} | Speed: {format_file_size(speed)}/s", end='', flush=True)
                    else:
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        print(f"\r📊 Downloaded: {format_file_size(downloaded)} | Speed: {format_file_size(speed)}/s", end='', flush=True)
        
        print()  # New line after progress
        elapsed = time.time() - start_time
        avg_speed = downloaded / elapsed if elapsed > 0 else 0
        
        print(f"✅ Download completed successfully!")
        print(f"📁 File saved as: {output_filename}")
        print(f"📏 Total size: {format_file_size(downloaded)}")
        print(f"⏱️  Time taken: {elapsed:.2f} seconds")
        print(f"🚀 Average speed: {format_file_size(avg_speed)}/s")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def format_file_size(bytes_size):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

def main():
    """Main function."""
    print("🎥 VideoXpert Export Downloader")
    print("=" * 50)
    
    # Check if custom URL is provided
    url = EXPORT_URL
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"🔗 Using custom URL: {url}")
    
    # Check if custom output filename is provided
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        print(f"📁 Using custom output file: {output_file}")
    
    print()
    
    # Download the export
    success = download_export(url, VIDEOXPERT_USERNAME, VIDEOXPERT_PASSWORD, output_file)
    
    if success:
        print("\n🎉 Export downloaded successfully!")
        sys.exit(0)
    else:
        print("\n💥 Export download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()