import os
import time
import logging
import cv2
from datetime import datetime
from .awsRecognitionService import AWSRekognitionService
from .personTracker import PersonTracker
from .movementLogger import MovementLogger

logger = logging.getLogger(__name__)

def start_process(camera, output_path):
    """
    Process video for analytics and movement tracking
    
    Args:
        camera: Camera configuration object from cameras.py
        output_path: Path to the video file to process
    
    Returns:
        dict: Processing results and statistics
    """
    try:
        logger.info(f"Starting video processing for camera: {camera}")
        
        # Initialize services
        aws_service = AWSRekognitionService()
        person_tracker = PersonTracker(camera_id=str(camera.get('id', 'unknown')))
        movement_logger = MovementLogger()
        
        # Always enable AWS for testing
        aws_enabled = True
        success, message = aws_service.enable_aws_rekognition()
        if not success:
            logger.warning(f"AWS Rekognition not enabled: {message}")
        aws_service.set_export_mode(True)
        logger.info("AWS Rekognition and export mode enabled by default for testing")
        
        # Use output_path as the video path (since that's what we're processing)
        video_path = output_path
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        
        # Setup video writer for processed output
        processed_output_path = output_path.replace('.mp4', '_processed.mp4')
        video_writer = None
        
        output_dir = os.path.dirname(processed_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # type: ignore
        video_writer = cv2.VideoWriter(processed_output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise Exception("Failed to create video writer")
        
        # Get stores configuration
        stores = camera.get('stores', {})
        
        # Processing statistics
        frame_count = 0
        aws_frame_skip = 60  # Process AWS face detection every 60th frame (like BlueprintTrack)
        processing_start_time = time.time()
        
        # Performance tracking
        timings = {
            'aws_detection': 0.0,
            'yolo_detection': 0.0,
            'tracking_update': 0.0,
            'movement_logging': 0.0
        }
        
        # Process frames
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_count / fps
                face_detections = []
                
                # AWS face detection every 60th frame (like BlueprintTrack)
                if frame_count % aws_frame_skip == 0 and aws_service.aws_enabled:
                    t_aws_start = time.time()
                    face_detections = aws_service.detect_faces(frame, current_time)
                    timings['aws_detection'] += time.time() - t_aws_start
                    if face_detections:
                        logger.info(f"Frame {frame_count}: Found {len(face_detections)} registered faces")
                else:
                    face_detections = []
                    
                # YOLO person detection
                t_yolo_start = time.time()
                detections = person_tracker.analyze_frame(frame)
                timings['yolo_detection'] += time.time() - t_yolo_start
                
                # Update tracking
                t_update_start = time.time()
                tracked_people = person_tracker.update(
                    detections['persons'], stores, frame_count, face_detections, frame
                )
                timings['tracking_update'] += time.time() - t_update_start
            
                # Draw bounding boxes and annotations on frame
                frame_with_annotations = frame.copy()
                
                # Draw face detection bounding boxes
                for face in face_detections:
                    bbox = face['bbox']
                    x, y, w, h = bbox
                    confidence = face.get('confidence', 0)
                    user_id = face.get('user_id', 'Unknown')
                    
                    # Draw face bounding box (red for recognized faces)
                    cv2.rectangle(frame_with_annotations, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Draw label with user ID and confidence
                    label = f"{user_id} ({confidence:.1f}%)"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame_with_annotations, (x, y - label_size[1] - 10), 
                                 (x + label_size[0], y), (0, 0, 255), -1)
                    cv2.putText(frame_with_annotations, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw person tracking bounding boxes
                for person_id, person in tracked_people.items():
                    bbox = person['bbox']
                    x, y, w, h = bbox
                    location = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
                    
                    # Get person identity (user_id and face_id)
                    user_id, face_id = person_tracker.get_person_identity(person_id)
                    
                    # Determine activity type
                    activity_type = 'walking' if person.get('is_moving', False) else 'standing'
                    
                    # Draw person bounding box (blue for tracked persons)
                    cv2.rectangle(frame_with_annotations, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Draw person ID and activity (enhanced like BlueprintTrack)
                    display_name = person_tracker.get_person_display_name(person_id)
                    person_label = f"{display_name} ({activity_type})"
                    label_size = cv2.getTextSize(person_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame_with_annotations, (x, y + h), 
                                 (x + label_size[0], y + h + label_size[1] + 10), (255, 0, 0), -1)
                    cv2.putText(frame_with_annotations, person_label, (x, y + h + label_size[1] + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Only log movement if we have a valid user_id (registered person)
                    if user_id:
                        movement_logger.log_person_movement(
                            person_id=user_id,  # Use actual user_id instead of tracking ID
                            location=location,
                            timestamp=datetime.now(),
                            camera_id=str(camera.get('id', 'unknown')),
                            confidence=person.get('confidence', 1.0),
                            store_id=person.get('current_store'),
                            activity_type=activity_type,
                            bbox=bbox,
                            face_id=face_id
                        )
                        logger.info(f"Logged movement for user {user_id} at location {location}")
                    else:
                        logger.debug(f"Skipping movement log for unregistered person {person_id}")
                
                # Write annotated frame to output video
                if video_writer and video_writer.isOpened():
                    video_writer.write(frame_with_annotations)
                
                frame_count += 1
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processing progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                frame_count += 1
                continue
        
        # Calculate processing statistics
        processing_time = time.time() - processing_start_time
        aws_calls = aws_service.api_calls_count
        tracked_persons = len(tracked_people)
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        
        # Results
        results = {
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'processing_time': processing_time,
            'processing_speed': frame_count / processing_time if processing_time > 0 else 0,
            'aws_api_calls': aws_calls,
            'tracked_persons': tracked_persons,
            'input_video_path': video_path,
            'processed_video_path': processed_output_path if video_writer else None,
            'performance_timings': timings,
            'aws_frame_skip': aws_frame_skip
        }
        
        logger.info(f"Video processing completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error in video processing: {e}")
        raise