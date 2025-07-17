from celery import shared_task
import redis
import json
import logging
from django.db import transaction
from django.conf import settings
from django.utils import timezone
from datetime import datetime
from .models import MovementLog
from core.models import UserMovement, Visit, User, MallStore
from wise_backend.logs.services.pelco_video import PelcoVideoService
from wise_backend.logs.services.cameras import cameras
from wise_backend.logs.services.videoProcessing import start_process
from celery import chord
from datetime import datetime, timedelta
import time
from celery.exceptions import MaxRetriesExceededError
import cv2
import numpy as np
import os
import boto3
from botocore.exceptions import ClientError
import torch
from ultralytics import YOLO
from shapely.geometry import Polygon

# PyTorch compatibility is now handled by using PyTorch 2.2.0
# No need for safe globals in this version

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=10, default_retry_delay=60)
def video_processing_task(self, output_path, camera):
    """Task to process video for a single camera"""
    try:
        logger.info(f"Processing video from {output_path}")
        
        # Call the start_process function to handle video processing
        results = start_process(camera, output_path)
        
        # # Clean up temporary files
        # try:
        #     # Delete the zip file if it exists
        #     if os.path.exists(output_path) and output_path.endswith('.zip'):
        #         os.remove(output_path)
        #         logger.info(f"Deleted zip file: {output_path}")
            
        #     # Delete the extracted video file if it was temporary
        #     video_path = camera.get('video_path')
        #     if video_path and os.path.exists(video_path) and '/tmp/' in video_path:
        #         os.remove(video_path)
        #         logger.info(f"Deleted temporary video file: {video_path}")
                
        # except Exception as cleanup_error:
        #     logger.warning(f"Error during cleanup: {cleanup_error}")
        
        return results
    except MaxRetriesExceededError:
        logging.error(
            f"Max retries exceeded for video_processing_task: "
            f"output_path={output_path}"
        )
        raise
    except Exception as e:
        logger.error(f"Error in video_processing_task: {e}")
        self.retry(exc=e)


@shared_task(bind=True, max_retries=10, default_retry_delay=60)
def export_camera_task(self, camera, start_time, end_time):
    """Task to export video for a single camera"""
    try:
        camera_id = camera['id']
        pelco_video_service = PelcoVideoService()
        export_id = pelco_video_service.export_async(camera_id, start_time, end_time)
        logger.info(f"Export result: {export_id}")
        
        if export_id is None:
            raise Exception("Export ID is None")
            
        # poll the export result every 10 seconds
        while True:
            
            export_result = pelco_video_service.get_export_status(export_id)
            if export_result == "Failed":
                raise Exception("Export failed")
            if export_result == 'Successful':
                # get the export url
                export_url = pelco_video_service.get_export_download_url(export_id)
                logger.info(f"Export URL: {export_url}")
                if export_url is None:
                    raise Exception("Export URL is None")
                # download the export
                output_path = pelco_video_service.download_export(export_url, f"exported_video_{camera_id}_{export_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")
                logger.info(f"Export downloaded to: {output_path}")
                # process the video
                video_processing_task.delay(output_path, camera)
                break
            time.sleep(10)
        
        return export_result
    except MaxRetriesExceededError:
        logging.error(
            f"Max retries exceeded for export_camera_task: "
            f"camera_id={camera_id}, start_time={start_time}, end_time={end_time}"
        )
        raise
    except Exception as e:
        logger.error(f"Error in export_camera_task: {e}")
        self.retry(exc=e)

@shared_task
def batch_camera_callback(results):
    """Callback after all camera exports are done"""
    logger.info(f"Batch camera export completed. Results: {results}")
    return results

@shared_task
def start_batch_camera():
    """Start a batch of cameras using Celery group/chord"""
    start_time = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    export_tasks = [
        export_camera_task.s(camera, start_time, end_time)
        for camera in cameras
    ]
    # Use chord to run all exports in parallel and then call the callback
    chord(export_tasks)(batch_camera_callback.s())