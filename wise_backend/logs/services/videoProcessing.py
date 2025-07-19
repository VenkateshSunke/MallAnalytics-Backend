import os
import time
import logging
import cv2
import numpy as np
import subprocess
import json
from datetime import datetime
from .awsRecognitionService import AWSRekognitionService
from .personTracker import PersonTracker
from .movementLogger import MovementLogger
from .blueprint_utils import get_blueprint_mapping_by_camera_id, get_blueprint_info_by_camera_id

logger = logging.getLogger(__name__)

def transform_blueprint_to_video_coordinates(blueprint_polygon, transformation_matrix):
    """
    Transform blueprint polygon coordinates to video coordinates using perspective transformation
    
    Args:
        blueprint_polygon: List of [x, y] coordinates in blueprint space
        transformation_matrix: 3x3 perspective transformation matrix
    
    Returns:
        List of [x, y] coordinates in video space
    """
    try:
        if not blueprint_polygon or len(blueprint_polygon) < 3:
            return None
        
        # Convert to numpy array and reshape for perspectiveTransform
        pts = np.array(blueprint_polygon, dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply perspective transformation
        video_pts = cv2.perspectiveTransform(pts, transformation_matrix)
        
        # Convert back to list of [x, y] coordinates
        video_polygon = [tuple(map(int, pt[0])) for pt in video_pts]
        
        return video_polygon
        
    except Exception as e:
        logger.error(f"Error transforming blueprint coordinates to video coordinates: {e}")
        return None

def draw_store_polygons(frame, stores, calibration_data=None):
    """
    Draw store polygons on the frame
    
    Args:
        frame: OpenCV frame to draw on
        stores: Dictionary of store configurations with polygon data
        calibration_data: Calibration data containing transformation matrices
    
    Returns:
        frame: Frame with store polygons drawn
    """
    frame_with_stores = frame.copy()
    
    for store_id, store in stores.items():
        # Get polygon data - try different possible keys
        polygon = store.get('video_polygon') or store.get('polygon')
        
        if not polygon or len(polygon) < 3:
            continue
            
        try:
            # If we have calibration data and transformation matrices, transform blueprint coordinates
            if calibration_data and 'store_matrices' in calibration_data:
                store_matrices = calibration_data['store_matrices']
                if store_id in store_matrices:
                    # Get the transformation matrix for this store
                    matrix = np.array(store_matrices[store_id], dtype=np.float32)
                    
                    # Transform blueprint coordinates to video coordinates
                    video_polygon = transform_blueprint_to_video_coordinates(polygon, matrix)
                    
                    if video_polygon:
                        polygon = video_polygon
                        logger.debug(f"Transformed polygon for store {store_id} using calibration matrix")
                    else:
                        logger.warning(f"Failed to transform polygon for store {store_id}")
                        continue
                else:
                    logger.warning(f"No transformation matrix found for store {store_id}")
                    continue
            else:
                # No calibration data, use polygon as-is (assume it's already in video coordinates)
                logger.debug(f"Using polygon as-is for store {store_id} (no calibration data)")
            
            # Convert polygon points to numpy array for drawing
            if isinstance(polygon[0], (list, tuple)):
                # Already in correct format [[x1,y1], [x2,y2], ...]
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            else:
                # Flat list format [x1,y1,x2,y2,...]
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            
            # Draw filled semi-transparent polygon
            overlay = frame_with_stores.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))  # Green fill
            cv2.addWeighted(overlay, 0.3, frame_with_stores, 0.7, 0, frame_with_stores)
            
            # Draw polygon outline
            cv2.polylines(frame_with_stores, [pts], True, (0, 255, 0), 2)
            
            # Draw store name at centroid
            centroid = np.mean(pts, axis=0).astype(int)[0]
            store_name = store.get('name', store_id)
            
            # Draw text background
            text_size = cv2.getTextSize(store_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame_with_stores, 
                         (centroid[0] - text_size[0]//2 - 5, centroid[1] - text_size[1] - 5),
                         (centroid[0] + text_size[0]//2 + 5, centroid[1] + 5),
                         (0, 0, 0), -1)
            
            # Draw store name
            cv2.putText(frame_with_stores, store_name, 
                       (centroid[0] - text_size[0]//2, centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            logger.debug(f"Drew polygon for store {store_name} with {len(polygon)} points")
            
        except Exception as e:
            logger.error(f"Error drawing polygon for store {store_id}: {e}")
            continue
    
    return frame_with_stores

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
        
        # Get video properties using ffprobe
        def get_video_info():
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-select_streams', 'v', video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                streams_info = json.loads(result.stdout)
                streams = streams_info.get('streams', [])
                
                # Use second video track if available, otherwise first
                if len(streams) > 1:
                    stream = streams[1]  # Use second track
                    stream_selector = '0:v:1'
                    logger.info(f"Using second video track for processing")
                elif len(streams) > 0:
                    stream = streams[0]  # Use first track  
                    stream_selector = '0:v:0'
                    logger.info(f"Using first video track for processing")
                else:
                    raise ValueError("No video streams found")
                
                width = int(stream.get('width', 640))
                height = int(stream.get('height', 480))
                
                # Get FPS
                fps_str = stream.get('r_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den != 0 else 30.0
                else:
                    fps = float(fps_str)
                
                # Get total frames
                if 'nb_frames' in stream:
                    total_frames = int(stream['nb_frames'])
                else:
                    duration = float(stream.get('duration', 0))
                    total_frames = int(duration * fps)
                
                return fps, width, height, total_frames, stream_selector
                
            except Exception as e:
                logger.error(f"Error getting video info: {e}")
                return 30.0, 640, 480, 0, '0:v:0'
        
        fps, width, height, total_frames, stream_selector = get_video_info()
        
        logger.info(f"Processing video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        
        # Setup FFmpeg reader process
        ffmpeg_read_cmd = [
            'ffmpeg', '-i', video_path,
            '-map', stream_selector,
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-vsync', '0', '-'
        ]
        ffmpeg_reader = subprocess.Popen(ffmpeg_read_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Setup video writer for processed output
        processed_output_path = output_path.replace('.mp4', '_processed.mp4')
        
        output_dir = os.path.dirname(processed_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Setup FFmpeg writer process
        ffmpeg_write_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', processed_output_path
        ]
        ffmpeg_writer = subprocess.Popen(ffmpeg_write_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Get blueprint mapping data based on camera ID
        camera_id = str(camera.get('id', 'unknown'))
        blueprint_mapping = get_blueprint_mapping_by_camera_id(camera_id)
        
        if blueprint_mapping:
            logger.info(f"Using blueprint mapping for camera {camera_id}")
            stores = blueprint_mapping.get('stores', {})
            calibration_data = blueprint_mapping.get('calibration', {})
            logger.info(f"Found {len(stores)} stores in blueprint mapping")
            if calibration_data and 'store_matrices' in calibration_data:
                logger.info(f"Found calibration data with {len(calibration_data['store_matrices'])} transformation matrices")
            else:
                logger.warning("No calibration data found in blueprint mapping")
        else:
            logger.warning(f"No blueprint mapping found for camera {camera_id}, using default stores config")
            stores = camera.get('stores', {})
            calibration_data = None
        
        # Track store entries to avoid duplicate logging
        store_entry_logged = {}  # person_id -> {store_id: timestamp}
        
        # Processing statistics
        frame_count = 0
        aws_frame_skip = 60  # Process AWS face detection every 60th frame (like BlueprintTrack)
        processing_start_time = time.time()
        frame_size = height * width * 3
        
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
                # Read frame from FFmpeg
                raw_frame = ffmpeg_reader.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((height, width, 3))
                
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
                
                # Draw store polygons FIRST (so they appear behind other elements)
                frame_with_annotations = draw_store_polygons(frame_with_annotations, stores, calibration_data)
                
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
                
                # Draw person tracking bounding boxes and handle store entries
                for person_id, person in tracked_people.items():
                    bbox = person['bbox']
                    x, y, w, h = bbox
                    location = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
                    
                    # Get person identity (user_id and face_id)
                    user_id, face_id = person_tracker.get_person_identity(person_id)
                    
                    # Determine activity type
                    activity_type = 'walking' if person.get('is_moving', False) else 'standing'
                    
                    # Check for store entry
                    current_store = person.get('current_store')
                    if current_store and current_store in stores:
                        # Check if this is a new store entry (not logged recently)
                        current_timestamp = datetime.now()
                        if (person_id not in store_entry_logged or 
                            current_store not in store_entry_logged[person_id] or
                            (current_timestamp - store_entry_logged[person_id][current_store]).seconds > 5):
                            
                            # Log store entry
                            store_name = stores[current_store].get('name', current_store)
                            logger.info(f"Person {person_id} entered store: {store_name}")
                            
                            # Initialize tracking for this person if needed
                            if person_id not in store_entry_logged:
                                store_entry_logged[person_id] = {}
                            
                            store_entry_logged[person_id][current_store] = current_timestamp
                            
                            # Log to movement logger if we have a valid user_id
                            if user_id:
                                movement_logger.log_person_movement(
                                    person_id=user_id,
                                    location=location,
                                    timestamp=current_timestamp,
                                    camera_id=camera_id,
                                    confidence=person.get('confidence', 1.0),
                                    store_id=current_store,
                                    activity_type='store_entered',
                                    bbox=bbox,
                                    face_id=face_id
                                )
                                logger.info(f"Logged store entry for user {user_id} into {store_name}")
                    
                    # Draw person bounding box (blue for tracked persons)
                    cv2.rectangle(frame_with_annotations, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Draw person ID and activity (enhanced like BlueprintTrack)
                    display_name = person_tracker.get_person_display_name(person_id)
                    person_label = f"{display_name} ({activity_type})"
                    if current_store:
                        store_name = stores[current_store].get('name', current_store)
                        person_label += f" in {store_name}"
                    
                    label_size = cv2.getTextSize(person_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame_with_annotations, (x, y + h), 
                                 (x + label_size[0], y + h + label_size[1] + 10), (255, 0, 0), -1)
                    cv2.putText(frame_with_annotations, person_label, (x, y + h + label_size[1] + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Log general movement if we have a valid user_id (registered person)
                    if user_id:
                        movement_logger.log_person_movement(
                            person_id=user_id,  # Use actual user_id instead of tracking ID
                            location=location,
                            timestamp=datetime.now(),
                            camera_id=camera_id,
                            confidence=person.get('confidence', 1.0),
                            store_id=person.get('current_store'),
                            activity_type=activity_type,
                            bbox=bbox,
                            face_id=face_id
                        )
                        logger.debug(f"Logged movement for user {user_id} at location {location}")
                    else:
                        logger.debug(f"Skipping movement log for unregistered person {person_id}")
                
                # Write annotated frame to FFmpeg writer
                ffmpeg_writer.stdin.write(frame_with_annotations.tobytes())
                
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
        ffmpeg_reader.stdout.close()
        ffmpeg_reader.wait()
        ffmpeg_writer.stdin.close()
        ffmpeg_writer.wait()
        
        # Results
        results = {
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'processing_time': processing_time,
            'processing_speed': frame_count / processing_time if processing_time > 0 else 0,
            'aws_api_calls': aws_calls,
            'tracked_persons': tracked_persons,
            'input_video_path': video_path,
            'processed_video_path': processed_output_path,
            'performance_timings': timings,
            'aws_frame_skip': aws_frame_skip,
            'camera_id': camera_id,
            'blueprint_used': bool(blueprint_mapping),
            'stores_processed': len(stores),
            'store_entries_logged': len(store_entry_logged),
            'calibration_available': bool(calibration_data and 'store_matrices' in calibration_data)
        }
        
        logger.info(f"Video processing completed: {results}")
        return results
        
    except Exception as e:
        print("Hello")
        logger.error(f"Error in video processing: {e}")
        raise