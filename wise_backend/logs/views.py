import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
import os
# from .tasks import (
#     add_movement_log_to_queue, 
#     get_queue_status, 
#     clear_movement_logs_queue,
#     test_celery_task,
#     process_movement_logs_batch,
#     start_visit,
#     end_visit,
# )
from .services.videoProcessing import start_process
from .models import MovementLog
from core.models import UserMovement, Visit, User
import logging
from django.utils import timezone
import ffmpeg

logger = logging.getLogger(__name__)


class StartVisitView(APIView):
    """
    POST /api/logs/start-visit/ - Start a new visit for a user
    """
    def post(self, request):
        try:
            user_id = request.data.get('user_id')
            if not user_id:
                return Response(
                    {'error': 'user_id is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            task = start_visit.delay(user_id)
            result = task.get(timeout=10)
            
            return Response({
                'message': result,
                'task_id': task.id
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error starting visit: {e}")
            return Response(
                {'error': 'Failed to start visit'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class EndVisitView(APIView):
    """
    POST /api/logs/end-visit/ - End the active visit for a user
    """
    def post(self, request):
        try:
            user_id = request.data.get('user_id')
            if not user_id:
                return Response(
                    {'error': 'user_id is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            task = end_visit.delay(user_id)
            result = task.get(timeout=10)
            
            return Response({
                'message': result,
                'task_id': task.id
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error ending visit: {e}")
            return Response(
                {'error': 'Failed to end visit'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AddMovementLogView(APIView):
    """
    POST /api/logs/add-movement/ - Add movement log to Redis queue
    ONLY processes movement logs when face_id (UUID) is provided for identified users
    """
    def post(self, request):
        try:
            data = request.data
            
            # Debug logging
            logger.info(f"Received movement data: {data}")
            
            # Check if face_id is provided - if not, ignore the request (return success but don't process)
            face_id_uuid = data.get('face_id')  # UUID like 689cb9ff-6d15-443e-a8f3-89ae0c150288
            
            if not face_id_uuid:
                # No face_id provided - this is anonymous tracking, return success but don't process
                logger.info(f"No face_id provided - ignoring anonymous tracking data for user_id: {data.get('user_id', 'unknown')}")
                return Response({
                    'message': 'No face_id provided - anonymous tracking ignored',
                    'processed': False,
                    'reason': 'Only identified users (with face_id) are tracked'
                }, status=status.HTTP_200_OK)
            
            # Check required fields for face_id processing
            if 'state' not in data:
                return Response(
                    {'error': 'Missing required field: state'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                # Look up existing user by face_id UUID - DO NOT CREATE NEW USERS
                user = User.objects.get(face_id=face_id_uuid)
                user_id = user.user_id  # This is the database user_id (e.g., U1YL9ZG5P)
                received_user_id = data.get('user_id')  # This is from face recognition (e.g., "Felipe")
                logger.info(f"Found existing user {user_id} ({user.name}) for face_id: {face_id_uuid}. Received user_id from script: {received_user_id}")
            except User.DoesNotExist:
                # Return error if face_id not found - users should already exist
                logger.warning(f"Face ID not found in database: {face_id_uuid}")
                return Response(
                    {'error': f'User with face_id {face_id_uuid} not found. Users must be registered first.'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Handle location data (support track_id/bbox format from video script)
            camera_id = data.get('camera_id', 'CAM_UNKNOWN')
            location = data.get('location')
            
            if not location:
                # Derive from track_id/bbox (video script format)
                track_id = data.get('track_id')
                if track_id:
                    location = f"Track_{track_id}"
                    
                    # If bbox is provided, use it for better location
                    bbox = data.get('bbox')
                    if bbox and isinstance(bbox, list) and len(bbox) >= 4:
                        x, y = bbox[0], bbox[1]
                        location = f"Area_({x},{y})"
                else:
                    location = 'Unknown_Location'
            
            # Queue the movement log for identified user (use database user_id, not face recognition name)
            # queue_task = add_movement_log_to_queue.delay(
            #     user_id=user_id,  # This is the correct database user_id (e.g., U1YL9ZG5P)
            #     camera_id=camera_id,
            #     location=location,
            #     state=data['state'],
            #     store=data.get('store'),
            #     timestamp=data.get('timestamp')
            # )
            
            # Wait for the queue task to complete before triggering batch processing
            try:
                queue_result = queue_task.get(timeout=5)  # Wait up to 5 seconds
                logger.info(f"Movement queued successfully: {queue_result}")
                
                # Now trigger batch processing
                process_task = process_movement_logs_batch.delay()
                logger.info(f"Triggered batch processing for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to queue movement or trigger batch processing: {e}")
            
            return Response({
                'message': 'Movement log queued and processed for identified user',
                'queue_task_id': queue_task.id,
                'user_id': user_id,
                'user_name': user.name,
                'face_id': face_id_uuid,
                'processed': True
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error queueing movement log: {e}")
            return Response(
                {'error': f'Failed to queue movement log: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class QueueStatusView(APIView):
    """
    GET /api/logs/queue-status/ - Get Redis queue status
    """
    def get(self, request):
        try:
            task = get_queue_status.delay()
            result = task.get(timeout=10)  # Wait up to 10 seconds for result
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return Response(
                {'error': 'Failed to get queue status'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ProcessLogsView(APIView):
    """
    POST /api/logs/process/ - Manually trigger log processing
    """
    def post(self, request):
        try:
            task = process_movement_logs_batch.delay()
            return Response({
                'message': 'Log processing started',
                'task_id': task.id
            }, status=status.HTTP_202_ACCEPTED)
        except Exception as e:
            logger.error(f"Error starting log processing: {e}")
            return Response(
                {'error': 'Failed to start log processing'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ClearQueueView(APIView):
    """
    POST /api/logs/clear-queue/ - Clear the movement logs queue
    """
    def post(self, request):
        try:
            task = clear_movement_logs_queue.delay()
            result = task.get(timeout=10)
            return Response({
                'message': result
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error clearing queue: {e}")
            return Response(
                {'error': 'Failed to clear queue'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TestCeleryView(APIView):
    """
    GET /api/logs/test-celery/ - Test Celery connection
    """
    def get(self, request):
        try:
            task = test_celery_task.delay()
            result = task.get(timeout=10)
            return Response({
                'message': result,
                'task_id': task.id,
                'celery_working': True
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Celery test failed: {e}")
            return Response({
                'error': 'Celery test failed',
                'celery_working': False,
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MovementLogListView(APIView):
    """
    GET /api/logs/movements/ - Get movement logs with pagination (from UserMovement table)
    """
    def get(self, request):
        try:
            # Get query parameters
            page = int(request.query_params.get('page', 1))
            page_size = int(request.query_params.get('page_size', 50))
            user_id = request.query_params.get('user_id')
            visit_id = request.query_params.get('visit_id')
            
            # Build queryset
            queryset = UserMovement.objects.select_related('visit__user', 'store').all().order_by('-start_time')
            
            if user_id:
                queryset = queryset.filter(visit__user__user_id=user_id)
            
            if visit_id:
                queryset = queryset.filter(visit__visit_id=visit_id)
            
            # Simple pagination
            start = (page - 1) * page_size
            end = start + page_size
            movements = queryset[start:end]
            
            # Serialize data
            data = []
            for movement in movements:
                data.append({
                    'movement_id': movement.movement_id,
                    'visit_id': movement.visit.visit_id,
                    'user_id': movement.visit.user.user_id,
                    'user_name': movement.visit.user.name,
                    'start_time': movement.start_time.isoformat() if movement.start_time else None,
                    'end_time': movement.end_time.isoformat() if movement.end_time else None,
                    'situation': movement.situation,
                    'camera_id': movement.camera_id,
                    'location': movement.location,
                    'store_code': movement.store.store_code if movement.store else None,
                    'store_name': movement.store.store_name if movement.store else None,
                })
            
            return Response({
                'results': data,
                'page': page,
                'page_size': page_size,
                'total_count': queryset.count()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching movement logs: {e}")
            return Response(
                {'error': 'Failed to fetch movement logs'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class VisitListView(APIView):
    """
    GET /api/logs/visits/ - Get visits with pagination
    """
    def get(self, request):
        try:
            # Get query parameters
            page = int(request.query_params.get('page', 1))
            page_size = int(request.query_params.get('page_size', 50))
            user_id = request.query_params.get('user_id')
            active_only = request.query_params.get('active_only', 'false').lower() == 'true'
            
            # Build queryset
            queryset = Visit.objects.select_related('user').all().order_by('-start_time')
            
            if user_id:
                queryset = queryset.filter(user__user_id=user_id)
            
            if active_only:
                queryset = queryset.filter(end_time__isnull=True)
            
            # Simple pagination
            start = (page - 1) * page_size
            end = start + page_size
            visits = queryset[start:end]
            
            # Serialize data
            data = []
            for visit in visits:
                # Count movements for this visit
                movement_count = UserMovement.objects.filter(visit=visit).count()
                
                data.append({
                    'visit_id': visit.visit_id,
                    'user_id': visit.user.user_id,
                    'user_name': visit.user.name,
                    'start_time': visit.start_time.isoformat() if visit.start_time else None,
                    'end_time': visit.end_time.isoformat() if visit.end_time else None,
                    'duration': str(visit.duration) if visit.duration else None,
                    'visit_date': visit.visit_date.isoformat() if visit.visit_date else None,
                    'stores_visited': visit.stores_visited,
                    'movement_count': movement_count,
                    'is_active': visit.end_time is None
                })
            
            return Response({
                'results': data,
                'page': page,
                'page_size': page_size,
                'total_count': queryset.count()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching visits: {e}")
            return Response(
                {'error': 'Failed to fetch visits'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SystemHealthView(APIView):
    """
    GET /api/logs/health/ - Get system health status (Redis + Celery)
    """
    def get(self, request):
        health_status = {
            'redis': False,
            'celery': False,
            'database': False,
            'overall_status': 'unhealthy'
        }
        
        # Test Redis
        try:
            task = get_queue_status.delay()
            redis_result = task.get(timeout=5)
            health_status['redis'] = redis_result.get('redis_connected', False)
        except:
            health_status['redis'] = False
        
        # Test Celery
        try:
            task = test_celery_task.delay()
            task.get(timeout=5)
            health_status['celery'] = True
        except:
            health_status['celery'] = False
        
        # Test Database
        try:
            UserMovement.objects.count()
            Visit.objects.count()
            health_status['database'] = True
        except:
            health_status['database'] = False
        
        # Overall status
        if all([health_status['redis'], health_status['celery'], health_status['database']]):
            health_status['overall_status'] = 'healthy'
        elif any([health_status['redis'], health_status['celery'], health_status['database']]):
            health_status['overall_status'] = 'partial'
        
        status_code = status.HTTP_200_OK if health_status['overall_status'] == 'healthy' else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return Response(health_status, status=status_code)


class UserVisitDetailView(APIView):
    """
    GET /api/logs/user-visit-detail/?face_id=<uuid> - Get detailed visit and movement info for a user
    """
    def get(self, request):
        try:
            face_id = request.query_params.get('face_id')
            user_id = request.query_params.get('user_id')
            
            if not face_id and not user_id:
                return Response(
                    {'error': 'Either face_id or user_id is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Find user
            user = None
            if face_id:
                try:
                    user = User.objects.get(face_id=face_id)
                except User.DoesNotExist:
                    return Response(
                        {'error': f'User with face_id {face_id} not found'}, 
                        status=status.HTTP_404_NOT_FOUND
                    )
            elif user_id:
                try:
                    user = User.objects.get(user_id=user_id)
                except User.DoesNotExist:
                    return Response(
                        {'error': f'User with user_id {user_id} not found'}, 
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            # Get today's visit
            today = timezone.now().date()
            today_visit = Visit.objects.filter(user=user, visit_date=today).first()
            
            # Get all visits for this user
            all_visits = Visit.objects.filter(user=user).order_by('-visit_date')[:10]
            
            # Get movements for today's visit
            today_movements = []
            if today_visit:
                movements = UserMovement.objects.filter(visit=today_visit).order_by('start_time')
                today_movements = [
                    {
                        'movement_id': m.movement_id,
                        'start_time': m.start_time,
                        'situation': m.situation,
                        'camera_id': m.camera_id,
                        'location': m.location,
                        'store': m.store.store_code if m.store else None
                    }
                    for m in movements
                ]
            
            return Response({
                'user': {
                    'user_id': user.user_id,
                    'name': user.name,
                    'face_id': user.face_id
                },
                'today_visit': {
                    'visit_id': today_visit.visit_id if today_visit else None,
                    'start_time': today_visit.start_time if today_visit else None,
                    'end_time': today_visit.end_time if today_visit else None,
                    'visit_date': today_visit.visit_date if today_visit else None,
                    'movements_count': len(today_movements)
                } if today_visit else None,
                'today_movements': today_movements,
                'recent_visits': [
                    {
                        'visit_id': v.visit_id,
                        'visit_date': v.visit_date,
                        'start_time': v.start_time,
                        'end_time': v.end_time,
                        'duration': str(v.duration) if v.duration else None,
                        'stores_visited': v.stores_visited
                    }
                    for v in all_visits
                ]
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting user visit details: {e}")
            return Response(
                {'error': f'Failed to get user visit details: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

def get_video_info(video_path):
    """
    Get comprehensive video information including available tracks using ffmpeg
    
    Args:
        video_path: Path to the video file
    
    Returns:
        dict: Video information including streams, tracks, duration, etc.
    """
    try:
        # Probe the video file
        probe = ffmpeg.probe(video_path)
        
        # Extract video streams
        video_streams = []
        audio_streams = []
        
        for i, stream in enumerate(probe['streams']):
            if stream['codec_type'] == 'video':
                # Get video stream info
                fps_parts = stream.get('r_frame_rate', '30/1').split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
                
                video_info = {
                    'track_index': i,
                    'stream_index': stream['index'],
                    'codec': stream.get('codec_name', 'unknown'),
                    'width': int(stream.get('width', 0)),
                    'height': int(stream.get('height', 0)),
                    'fps': round(fps, 2),
                    'duration': float(stream.get('duration', 0)),
                    'nb_frames': int(stream.get('nb_frames', 0)),
                    'pixel_format': stream.get('pix_fmt', 'unknown'),
                    'profile': stream.get('profile', 'unknown'),
                    'level': stream.get('level', 'unknown')
                }
                
                # Add bitrate if available
                if 'bit_rate' in stream:
                    video_info['bitrate'] = int(stream['bit_rate'])
                
                # Add display aspect ratio if available
                if 'display_aspect_ratio' in stream:
                    video_info['aspect_ratio'] = stream['display_aspect_ratio']
                
                video_streams.append(video_info)
                
            elif stream['codec_type'] == 'audio':
                # Get audio stream info
                audio_info = {
                    'stream_index': stream['index'],
                    'codec': stream.get('codec_name', 'unknown'),
                    'sample_rate': int(stream.get('sample_rate', 0)),
                    'channels': int(stream.get('channels', 0)),
                    'duration': float(stream.get('duration', 0)),
                    'channel_layout': stream.get('channel_layout', 'unknown')
                }
                
                if 'bit_rate' in stream:
                    audio_info['bitrate'] = int(stream['bit_rate'])
                
                audio_streams.append(audio_info)
        
        # Get format information
        format_info = probe.get('format', {})
        
        # Calculate total duration from format if not available in streams
        total_duration = float(format_info.get('duration', 0))
        if total_duration == 0 and video_streams:
            total_duration = max(stream.get('duration', 0) for stream in video_streams)
        
        # Update video streams with container duration if they have zero duration
        for stream in video_streams:
            if stream.get('duration', 0) == 0 and total_duration > 0:
                stream['duration'] = total_duration
                logger.info(f"Updated track {stream.get('track_index')} duration from 0 to {total_duration} (using container duration)")
            
            # Estimate frame count if nb_frames is zero but we have duration and fps
            if stream.get('nb_frames', 0) == 0 and stream.get('duration', 0) > 0 and stream.get('fps', 0) > 0:
                estimated_frames = int(stream['duration'] * stream['fps'])
                stream['nb_frames'] = estimated_frames
                stream['estimated_frames'] = True  # Flag to indicate this is an estimate
                logger.info(f"Estimated track {stream.get('track_index')} frames: {estimated_frames} (duration={stream['duration']}s * fps={stream['fps']})")
        
        # Build comprehensive video info
        video_info = {
            'filename': format_info.get('filename', video_path),
            'format_name': format_info.get('format_name', 'unknown'),
            'format_long_name': format_info.get('format_long_name', 'unknown'),
            'duration': total_duration,
            'size': int(format_info.get('size', 0)),
            'bitrate': int(format_info.get('bit_rate', 0)) if 'bit_rate' in format_info else None,
            'video_streams': video_streams,
            'audio_streams': audio_streams,
            'total_streams': len(probe['streams']),
            'video_tracks_count': len(video_streams),
            'audio_tracks_count': len(audio_streams)
        }
        
        return video_info
        
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error while probing video: {e}")
        return None
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return None

class TestVideoProcessingView(APIView):
    """
    POST /api/logs/test-video-processing/ - Test video processing function with ffmpeg support
    """
    def post(self, request):
        logger.info("TestVideoProcessingView.post() called")
        try:
            # Get parameters from request
            video_path = request.data.get('video_path')
            camera_id = request.data.get('camera_id', 'test_camera_001')
            camera_name = request.data.get('camera_name', 'Test Camera')
            video_track_index = request.data.get('video_track_index', 0)  # Default to first track

            logger.info(f"Received parameters: video_path={video_path}, camera_id={camera_id}, track_index={video_track_index}")

            if not video_path:
                return Response({'error': 'video_path is required'}, status=status.HTTP_400_BAD_REQUEST)

            # Validate video file
            if not os.path.exists(video_path):
                return Response({'error': f'Video file not found: {video_path}'}, status=status.HTTP_404_NOT_FOUND)

            # Get stream info to validate track
            video_info = get_video_info(video_path)
            
            # More lenient validation for corrupted files - check if tracks exist and have basic info
            valid_tracks = []
            for s in video_info.get('video_streams', []):
                # Check if track has basic video properties (width, height, codec)
                has_basic_info = (
                    int(s.get('width', 0)) > 0 and 
                    int(s.get('height', 0)) > 0 and 
                    s.get('codec', '') != 'unknown'
                )
                
                # For corrupted files, accept tracks with basic info even if duration/frames are 0
                if has_basic_info:
                    valid_tracks.append(s)
                    logger.info(f"Track {s.get('track_index')} accepted (basic info present, duration={s.get('duration')}, frames={s.get('nb_frames')})")

            if not valid_tracks:
                return Response({'error': 'No valid video tracks found (missing basic video properties)'}, status=status.HTTP_400_BAD_REQUEST)

            # If user provided invalid track index, fall back to first valid one
            if video_track_index >= len(video_info['video_streams']):
                logger.warning(f"Track index {video_track_index} out of bounds, falling back to 0")
                video_track_index = 0

            selected_track = video_info['video_streams'][video_track_index]
            if selected_track.get('duration', 0) == 0 or selected_track.get('nb_frames', 0) == 0:
                logger.warning(f"Selected track {video_track_index} has zero duration or frames. This is acceptable for corrupted files.")
                # Don't change the track index - let the processing handle it
                # The error tolerance parameters in videoProcessing.py should handle corrupted streams

            # Create test camera config
            camera_config = {
                'id': camera_id,
                'name': camera_name,
                'selected_tracks': [1, 2],
                'aws_enabled': True,
                'stores': {
                    'store_001': {
                        'name': 'Test Store 1',
                        'video_polygon': [[100, 100], [300, 100], [300, 300], [100, 300]],
                        'is_mapped': True
                    },
                    'store_002': {
                        'name': 'Test Store 2',
                        'video_polygon': [[400, 100], [600, 100], [600, 300], [400, 300]],
                        'is_mapped': True
                    }
                }
            }

            logger.info("About to call start_process()...")
            results = start_process(camera_config, video_path, video_track_index)
            logger.info("start_process completed successfully")

            return Response({
                'message': 'Video processing completed successfully',
                'results': results,
                'camera_config': camera_config
            }, status=status.HTTP_200_OK)

        except FileNotFoundError as e:
            logger.error(f"Video file not found: {e}")
            return Response({'error': f'Video file not found: {str(e)}'}, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return Response({'error': f'Video processing failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class VideoInfoView(APIView):
    """
    POST /api/logs/video-info/ - Get video information and available tracks
    """
    def post(self, request):
        try:
            video_path = request.data.get('video_path')
            
            if not video_path:
                return Response(
                    {'error': 'video_path is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if not os.path.exists(video_path):
                return Response(
                    {'error': f'Video file not found: {video_path}'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get video information using ffmpeg
            video_info = get_video_info(video_path)
            
            if not video_info:
                return Response(
                    {'error': 'Could not read video information'}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            return Response({
                'message': 'Video information retrieved successfully',
                'video_info': video_info,
                'video_path': video_path
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return Response(
                {'error': f'Failed to get video information: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        