from django.conf import settings
from datetime import datetime
import requests
import urllib3
import base64

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MALL1_PELCO_VIDEO_API_HOST = settings.MALL1_PELCO_VIDEO_API_HOST


class PelcoVideoService:
    def __init__(self):
        self.base_url = MALL1_PELCO_VIDEO_API_HOST
        
    def health_check(self):
        url = f"{self.base_url}/api/health"
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                return True
        return False

    def export_async(self, camera_id: str, start_time: str, end_time: str) -> str | None:
        url = f"{self.base_url}/api/exports/async"
        # Format datetimes as "YYYY-MM-DD HH:MM:SS"
        response = requests.post(url, json={
            "cameraId": camera_id,
            "startTime": start_time,
            "endTime": end_time
        }, verify=False)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return data.get("exportId")
            else:
                raise Exception(data.get("error"))
        else:
            raise Exception(response.text)

    def get_export_status(self, export_id: str) -> str | None:
        url = f"{self.base_url}/api/exports/{export_id}/status"
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            data = response.json()
            return data.get("status")
        return None

    def get_export_download_url(self, export_id: str) -> str | None:
        url = f"{self.base_url}/api/exports/{export_id}/download-url"
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            data = response.json()
            return data.get("downloadUrl")
        return None
    
    def encode_auth_headers(self, username, password):
        """Encode username and password for VideoXpert authentication headers."""
        user_b64 = base64.b64encode(username.encode('ascii')).decode('ascii')
        pass_b64 = base64.b64encode(password.encode('ascii')).decode('ascii')
        return user_b64, pass_b64
    
    def download_export(self, export_url: str, output_path: str = None):
        """
        Downloads the export zip file from VideoXpert, extracts it to find the MKV file,
        cleans up all other files and the zip, and returns the MKV file path.
        Uses X-Serenity-User and X-Serenity-Password headers with base64-encoded credentials.
        Handles very large files by streaming and writing in chunks.
        Returns the MKV file path if successful, else None.
        """
        import os
        import tempfile
        import zipfile
        import shutil
        import glob

        chunk_size = 1024 * 1024  # 1MB per chunk

        # Create a temporary directory for extraction
        temp_dir = tempfile.mkdtemp(prefix="pelco_export_", dir="/tmp")
        
        # Generate zip file path
        if output_path is None:
            zip_filename = f"exported_video_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"
        else:
            zip_filename = output_path if output_path.endswith('.zip') else f"{output_path}.zip"
        
        zip_path = os.path.join(temp_dir, zip_filename)

        try:
            # Get VideoXpert credentials from settings
            username = settings.MALL1_PELCO_VIDEO_API_USER
            password = settings.MALL1_PELCO_VIDEO_API_PASSWORD
            
            # Encode authentication headers
            user_b64, pass_b64 = self.encode_auth_headers(username, password)
            
            # Set up VideoXpert authentication headers
            headers = {
                'X-Serenity-User': user_b64,
                'X-Serenity-Password': pass_b64,
                'User-Agent': 'VideoXpert-Export-Downloader/1.0'
            }

            # Download the zip file
            response = requests.get(export_url, stream=True, verify=False, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Download failed with status {response.status_code}: {response.text}")
            
            # Save the zip file
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            
            # Extract the zip file
            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find MKV files in the extracted directory (recursive search)
            mkv_files = []
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith('.mkv'):
                        mkv_files.append(os.path.join(root, file))
            
            if not mkv_files:
                raise Exception("No MKV file found in the exported zip")
            
            if len(mkv_files) > 1:
                # If multiple MKV files, use the largest one (usually the main video)
                mkv_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
            
            # Get the MKV file (first/largest one)
            mkv_file = mkv_files[0]
            
            # Create final MKV file path in /tmp
            mkv_filename = f"exported_video_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mkv"
            final_mkv_path = os.path.join("/tmp", mkv_filename)
            
            # Move the MKV file to final location
            shutil.move(mkv_file, final_mkv_path)
            
            # Clean up: remove the temporary directory (zip file and all extracted files)
            shutil.rmtree(temp_dir)
            
            return final_mkv_path
            
        except Exception as e:
            # Clean up temporary directory in case of error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e