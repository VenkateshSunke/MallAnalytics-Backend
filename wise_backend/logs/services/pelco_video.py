from django.conf import settings
from datetime import datetime
import requests

MALL1_PELCO_VIDEO_API_HOST = settings.MALL1_PELCO_VIDEO_API_HOST


class PelcoVideoService:
    def __init__(self):
        self.base_url = MALL1_PELCO_VIDEO_API_HOST
        
    def health_check(self):
        url = f"{self.base_url}/api/health"
        response = requests.get(url)
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
        })
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
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("status")
        return None

    def get_export_download_url(self, export_id: str) -> str | None:
        url = f"{self.base_url}/api/exports/{export_id}/download-url"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("downloadUrl")
        return None
    
    def download_export(self, export_url: str, output_path: str = None):
        """
        Downloads the export zip file from the given URL and saves it to a specified directory
        outside the local repo folder (e.g., /tmp or a configurable export directory).
        Handles very large files by streaming and writing in chunks.
        Returns the local file path if successful, else None.
        """
        import os
        import tempfile

        chunk_size = 1024 * 1024  # 1MB per chunk

        # Use /tmp or a configurable directory for output
        if output_path is None:
            # Use a unique temp file in /tmp
            fd, temp_path = tempfile.mkstemp(prefix="exported_video_", suffix=".zip", dir="/tmp")
            os.close(fd)  # We'll open it again below
            output_path = temp_path
        else:
            # If output_path is a relative path, put it in /tmp
            if not os.path.isabs(output_path):
                output_path = os.path.join("/tmp", output_path)

        response = requests.get(export_url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            return output_path
        return None