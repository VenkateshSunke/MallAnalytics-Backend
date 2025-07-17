from celery import shared_task
import redis
import json
import logging
from django.db import transaction
from django.conf import settings
from django.utils import timezone
from datetime import datetime, date
from .models import MovementLog, DailyVideoExport
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
        
        # Clean up temporary MKV file
        try:
            # Delete the MKV file if it exists
            if os.path.exists(output_path) and output_path.endswith('.mkv'):
                os.remove(output_path)
                logger.info(f"Deleted MKV file: {output_path}")
            elif os.path.exists(output_path):
                # Delete any other video file format
                os.remove(output_path)
                logger.info(f"Deleted video file: {output_path}")
                
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")
        
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
def export_camera_task(self, camera, start_time, end_time, export_date):
    """Task to export video for a single camera"""
    try:
        camera_id = camera['id']
        
        # Update export status to in_progress
        try:
            export_record = DailyVideoExport.objects.get(
                camera_id=camera_id, 
                export_date=export_date
            )
            export_record.status = 'in_progress'
            export_record.save()
        except DailyVideoExport.DoesNotExist:
            logger.warning(f"Export record not found for camera {camera_id} on {export_date}")
            return "Export record not found"
        
        pelco_video_service = PelcoVideoService()
        export_id = pelco_video_service.export_async(camera_id, start_time, end_time)
        logger.info(f"Export result: {export_id}")
        
        if export_id is None:
            export_record.status = 'failed'
            export_record.error_message = "Export ID is None"
            export_record.save()
            raise Exception("Export ID is None")
        
        # Save the export ID
        export_record.export_id = export_id
        export_record.save()
            
        # poll the export result every 10 seconds
        while True:
            export_result = pelco_video_service.get_export_status(export_id)
            if export_result == "Failed":
                export_record.status = 'failed'
                export_record.error_message = "Export failed"
                export_record.save()
                raise Exception("Export failed")
            if export_result == 'Successful':
                # get the export url
                export_url = pelco_video_service.get_export_download_url(export_id)
                logger.info(f"Export URL: {export_url}")
                if export_url is None:
                    export_record.status = 'failed'
                    export_record.error_message = "Export URL is None"
                    export_record.save()
                    raise Exception("Export URL is None")
                
                # Save the export URL
                export_record.export_url = export_url
                export_record.save()
                
                # download the export
                output_path = pelco_video_service.download_export(export_url, f"exported_video_{camera_id}_{export_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")
                logger.info(f"Export downloaded to: {output_path}")
                
                # Mark as completed
                export_record.status = 'completed'
                export_record.save()
                
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
        # Mark as failed
        try:
            export_record = DailyVideoExport.objects.get(
                camera_id=camera_id, 
                export_date=export_date
            )
            export_record.status = 'failed'
            export_record.error_message = "Max retries exceeded"
            export_record.save()
        except:
            pass
        raise
    except Exception as e:
        logger.error(f"Error in export_camera_task: {e}")
        # Mark as failed
        try:
            export_record = DailyVideoExport.objects.get(
                camera_id=camera_id, 
                export_date=export_date
            )
            export_record.status = 'failed'
            export_record.error_message = str(e)
            export_record.save()
        except:
            pass
        self.retry(exc=e)

@shared_task
def batch_camera_callback(results):
    """Callback after all camera exports are done"""
    logger.info(f"Batch camera export completed. Results: {results}")
    return results

@shared_task
def start_batch_camera():
    """Start a batch of cameras using Celery group/chord - idempotent version"""
    # Calculate the date for yesterday (the day we want to export)
    yesterday = (datetime.now() - timedelta(days=1)).date()
    
    # Calculate start and end times for business hours (10 hours of peak activity)
    # start_time = (datetime.now() - timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    # end_time = (datetime.now() - timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    
    # For testing: export only 10 minutes (from 9:00 to 9:10)
    start_time = (datetime.now() - timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    end_time = (datetime.now() - timedelta(days=1)).replace(hour=9, minute=10, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting batch camera export for date: {yesterday}")
    logger.info(f"Export time range: {start_time} to {end_time}")
    
    export_tasks = []
    
    for camera in cameras:
        camera_id = camera['id']
        
        # Check if export already exists for this camera and date
        export_record, created = DailyVideoExport.objects.get_or_create(
            camera_id=camera_id,
            export_date=yesterday,
            defaults={'status': 'pending'}
        )
        
        if created:
            logger.info(f"Created new export record for camera {camera_id} on {yesterday}")
            export_tasks.append(export_camera_task.s(camera, start_time, end_time, yesterday))
        elif export_record.status == 'pending':
            logger.info(f"Found pending export for camera {camera_id} on {yesterday}, adding to queue")
            export_tasks.append(export_camera_task.s(camera, start_time, end_time, yesterday))
        elif export_record.status == 'failed':
            logger.info(f"Found failed export for camera {camera_id} on {yesterday}, retrying")
            export_record.status = 'pending'
            export_record.error_message = None
            export_record.save()
            export_tasks.append(export_camera_task.s(camera, start_time, end_time, yesterday))
        else:
            logger.info(f"Export for camera {camera_id} on {yesterday} already {export_record.status}, skipping")
    
    if export_tasks:
        logger.info(f"Starting {len(export_tasks)} export tasks")
        # Use chord to run all exports in parallel and then call the callback
        chord(export_tasks)(batch_camera_callback.s())
    else:
        logger.info("No new exports to start - all cameras already processed for this date")
        return "No new exports needed"