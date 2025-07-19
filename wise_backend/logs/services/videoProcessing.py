import os
import time
import logging
import cv2
import numpy as np
import ffmpeg
from datetime import datetime
from .awsRecognitionService import AWSRekognitionService
from .personTracker import PersonTracker
from .movementLogger import MovementLogger
from .blueprint_utils import get_blueprint_mapping_by_camera_id, get_blueprint_info_by_camera_id

logger = logging.getLogger(__name__)

def get_video_info(video_path):
    """
    Get video information including available tracks using ffmpeg
    
    Args:
        video_path: Path to the video file
    
    Returns:
        dict: Video information including streams and tracks
    """
    try:
        probe = ffmpeg.probe(video_path)
        
        video_info = {
            'format': probe.get('format', {}),
            'streams': [],
            'video_tracks': [],
            'audio_tracks': [],
            'subtitle_tracks': []
        }
        
        for stream in probe.get('streams', []):
            stream_info = {
                'index': stream.get('index'),
                'codec_type': stream.get('codec_type'),
                'codec_name': stream.get('codec_name'),
                'width': stream.get('width'),
                'height': stream.get('height'),
                'duration': stream.get('duration'),
                'bit_rate': stream.get('bit_rate'),
                'language': stream.get('tags', {}).get('language', 'unknown')
            }
            
            video_info['streams'].append(stream_info)
            
            # Categorize streams by type
            if stream.get('codec_type') == 'video':
                video_info['video_tracks'].append(stream_info)
            elif stream.get('codec_type') == 'audio':
                video_info['audio_tracks'].append(stream_info)
            elif stream.get('codec_type') == 'subtitle':
                video_info['subtitle_tracks'].append(stream_info)
        
        logger.info(f"Video info for {video_path}: {len(video_info['video_tracks'])} video tracks, "
                   f"{len(video_info['audio_tracks'])} audio tracks, "
                   f"{len(video_info['subtitle_tracks'])} subtitle tracks")
        
        return video_info
        
    except Exception as e:
        logger.error(f"Error getting video info for {video_path}: {e}")
        return None

def create_ffmpeg_input(video_path, stream_index=0):
    """
    Create ffmpeg input stream for a specific video stream
    
    Args:
        video_path: Path to the video file
        stream_index: Index of the video stream to use (default: 0)
    
    Returns:
        ffmpeg input stream
    """
    try:
        # Create input with specific video stream
        if stream_index > 0:
            # Use map to select specific video stream by its actual index
            # stream_index is the actual stream index from the file (1, 2, etc.)
            # We need to map to the specific stream index
            video_stream = ffmpeg.input(video_path, map=f'0:{stream_index}')
            return video_stream
        else:
            # Use default video stream - return the full input, not just video
            input_stream = ffmpeg.input(video_path)
            return input_stream
            
    except Exception as e:
        logger.error(f"Error creating ffmpeg input for stream {stream_index}: {e}")
        return None

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

def start_process(camera, output_path, video_track_index=0):
    """
    Process video for analytics and movement tracking using ffmpeg
    
    Args:
        camera: Camera configuration object from cameras.py
        output_path: Path to the video file to process
        video_track_index: Index of the video track to process (default: 0)
    
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
        if not video_path:
            raise ValueError("Video path is required")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not os.path.isfile(video_path):
            raise ValueError(f"Path is not a file: {video_path}")
        
        logger.info(f"Processing video file: {video_path}")
        logger.info(f"File size: {os.path.getsize(video_path)} bytes")
        
        # Get video information using ffmpeg
        video_info = get_video_info(video_path)
        if not video_info:
            raise ValueError(f"Could not get video information for: {video_path}")
        
        # Check if requested track exists
        available_video_tracks = video_info.get('video_tracks', [])
        if not available_video_tracks:
            raise ValueError(f"No video tracks found in video file: {video_path}")
        
        # Map the requested track index to the actual stream index
        # video_track_index is 0-based for the video tracks array
        # but we need the actual stream index from the track
        if video_track_index >= len(available_video_tracks):
            logger.warning(f"Requested video track {video_track_index} not available. "
                          f"Available tracks: {len(available_video_tracks)} (indices 0-{len(available_video_tracks)-1}). Using track 0.")
            video_track_index = 0
        
        selected_track = available_video_tracks[video_track_index]
        actual_stream_index = selected_track.get('index')  # This is the real stream index (1, 2, etc.)
        
        logger.info(f"Using video track {video_track_index} (stream index {actual_stream_index}): {selected_track}")
        
        # Validate track has required properties
        if not selected_track.get('width') or not selected_track.get('height'):
            logger.warning(f"Video track {video_track_index} missing width/height, using defaults")
            selected_track['width'] = selected_track.get('width', 1920)
            selected_track['height'] = selected_track.get('height', 1080)
        
        # Get video properties from selected track
        width = selected_track.get('width', 1920)
        height = selected_track.get('height', 1080)
        
        # Get duration and calculate fps
        duration = 0.0
        try:
            track_duration = selected_track.get('duration')
            if track_duration is not None:
                duration = float(track_duration)
        except (ValueError, TypeError):
            duration = 0.0
            
        format_info = video_info.get('format', {})
        format_duration = 0.0
        try:
            format_duration = float(format_info.get('duration', 0))
        except (ValueError, TypeError):
            format_duration = 0.0
            
        total_frames = int(format_duration * 30) if format_duration > 0 else 3600  # Estimate frames
        fps = 30  # Default fps, will be updated if available
        
        # Try to get fps from format tags
        if 'tags' in format_info:
            tags = format_info['tags']
            if 'DURATION' in tags:
                # Parse duration to get more accurate frame count
                duration_str = tags['DURATION']
                # Duration format: HH:MM:SS.microseconds
                try:
                    time_parts = duration_str.split(':')
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    seconds = float(time_parts[2])
                    parsed_duration = hours * 3600 + minutes * 60 + seconds
                    if parsed_duration > 0:
                        duration = parsed_duration
                        total_frames = int(duration * fps)
                except (ValueError, IndexError, TypeError) as e:
                    logger.warning(f"Could not parse duration from tags: {e}")
                    pass
        
        logger.info(f"Processing video track {video_track_index}: {width}x{height} @ {fps:.1f}fps, "
                   f"estimated {total_frames} frames, duration: {duration:.2f}s")
        
        # Create ffmpeg input stream for the selected track
        ffmpeg_input = create_ffmpeg_input(video_path, actual_stream_index)
        if not ffmpeg_input:
            raise ValueError(f"Could not create ffmpeg input for track {video_track_index} (stream index {actual_stream_index})")
        
        # Setup ffmpeg process to read frames
        try:
            logger.info("Starting ffmpeg process...")
            
            # Ensure we have a valid input stream
            if not ffmpeg_input:
                raise ValueError(f"Invalid ffmpeg input stream for track {video_track_index}")
            
            # Create the ffmpeg command
            ffmpeg_process = (
                ffmpeg_input
                .output('pipe:', format='rawvideo', pix_fmt='bgr24', vsync='0', loglevel='error')
                .overwrite_output()
                .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
            )
            
            if not ffmpeg_process:
                raise ValueError(f"Could not start ffmpeg process for track {video_track_index}")
            
            # Give ffmpeg a moment to start
            time.sleep(0.5)
            
            # Check if process is still running
            if ffmpeg_process.poll() is not None:
                # Process terminated immediately, get error output
                try:
                    stderr_output = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                    raise ValueError(f"FFmpeg process terminated immediately. Error: {stderr_output}")
                except:
                    raise ValueError(f"FFmpeg process terminated immediately for track {video_track_index}")
            
            logger.info("FFmpeg process started successfully")
            
        except Exception as e:
            logger.error(f"Error starting ffmpeg process: {e}")
            logger.warning("Falling back to OpenCV for video reading")
            
            # Fallback to OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file with OpenCV: {video_path}")
            
            # Get video properties from OpenCV
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Using OpenCV fallback: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
            
            # Set ffmpeg_process to None to indicate we're using OpenCV
            ffmpeg_process = None
        
        # Setup video writer for processed output
        # Create a proper output path with _processed suffix
        base_name = os.path.splitext(output_path)[0]
        processed_output_path = f"{base_name}_processed.mp4"
        video_writer = None
        
        # Ensure output directory exists
        output_dir = os.path.dirname(processed_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Creating video writer for output: {processed_output_path}")
        logger.info(f"Video dimensions: {width}x{height}, FPS: {fps}")
        
        try:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # type: ignore
            video_writer = cv2.VideoWriter(processed_output_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                raise Exception(f"Failed to create video writer for {processed_output_path}")
            
            logger.info("Video writer created successfully")
            
        except Exception as e:
            logger.error(f"Error creating video writer: {e}")
            # Try alternative codec if mp4v fails
            try:
                logger.info("Trying alternative codec (XVID)")
                fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # type: ignore
                processed_output_path = f"{base_name}_processed.avi"
                video_writer = cv2.VideoWriter(processed_output_path, fourcc, fps, (width, height))
                
                if not video_writer.isOpened():
                    raise Exception(f"Failed to create video writer with XVID codec")
                
                logger.info("Video writer created successfully with XVID codec")
                
            except Exception as e2:
                logger.error(f"Error creating video writer with XVID codec: {e2}")
                logger.warning("Continuing without video output - processing will still work but no output video will be created")
                video_writer = None
                processed_output_path = None
        
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
        
        # Initialize tracking variables
        tracked_people = {}  # Initialize empty dict to avoid reference error
        
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
        frames_processed = 0
        
        if ffmpeg_process:
            # Use ffmpeg for frame reading
            frame_size = width * height * 3  # 3 bytes per pixel (BGR)
            logger.info(f"Starting frame processing with ffmpeg. Frame size: {frame_size} bytes")
            logger.info(f"Expected frame dimensions: {width}x{height}")
            
            while True:
                try:
                    # Read raw frame data from ffmpeg
                    raw_frame = ffmpeg_process.stdout.read(frame_size)
                    
                    # Check if we got a complete frame
                    if not raw_frame:
                        logger.info("No more frames to read from ffmpeg")
                        break
                        
                    if len(raw_frame) != frame_size:
                        logger.warning(f"Incomplete frame read: {len(raw_frame)} bytes instead of {frame_size}")
                        # Try to read more data to complete the frame
                        remaining_bytes = frame_size - len(raw_frame)
                        additional_data = ffmpeg_process.stdout.read(remaining_bytes)
                        if additional_data:
                            raw_frame += additional_data
                            if len(raw_frame) != frame_size:
                                logger.warning(f"Still incomplete frame after reading additional data: {len(raw_frame)} bytes")
                                break
                        else:
                            break
                    
                    # Convert raw bytes to numpy array
                    try:
                        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3)
                    except ValueError as e:
                        logger.error(f"Error reshaping frame data: {e}")
                        logger.error(f"Expected shape: ({height}, {width}, 3), got {len(raw_frame)} bytes")
                        break
                    
                    # Check if ffmpeg process is still running
                    if ffmpeg_process.poll() is not None:
                        logger.warning("FFmpeg process terminated unexpectedly")
                        break
                    
                    frames_processed += 1
                    
                    # Process the frame
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
                    
                    # Write annotated frame to output video
                    if video_writer and video_writer.isOpened():
                        try:
                            video_writer.write(frame_with_annotations)
                        except Exception as e:
                            logger.error(f"Error writing frame {frame_count}: {e}")
                            # Continue processing even if frame writing fails
                    
                    frame_count += 1
                    
                    # Log progress every 100 frames
                    if frame_count % 100 == 0:
                        progress = (frame_count / total_frames) * 100
                        logger.info(f"Processing progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                        
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
                    frame_count += 1
                    continue
        else:
            # Use OpenCV for frame reading (fallback)
            logger.info("Using OpenCV for frame reading")
            
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frames_processed += 1
                    
                    # Process the frame (same logic as above)
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
                    
                    # Write annotated frame to output video
                    if video_writer and video_writer.isOpened():
                        try:
                            video_writer.write(frame_with_annotations)
                        except Exception as e:
                            logger.error(f"Error writing frame {frame_count}: {e}")
                            # Continue processing even if frame writing fails
                    
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
        
        logger.info(f"Frame processing completed. Frames processed: {frames_processed}")
        
        if frames_processed == 0:
            logger.warning("No frames were processed - this might indicate an issue with the video file or ffmpeg setup")
        
        # Cleanup
        if ffmpeg_process:
            ffmpeg_process.stdout.close()
            ffmpeg_process.stderr.close()
            ffmpeg_process.wait()
        elif 'cap' in locals():
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
            'processed_video_path': processed_output_path,
            'performance_timings': timings,
            'aws_frame_skip': aws_frame_skip,
            'camera_id': camera_id,
            'blueprint_used': bool(blueprint_mapping),
            'stores_processed': len(stores),
            'store_entries_logged': len(store_entry_logged),
            'calibration_available': bool(calibration_data and 'store_matrices' in calibration_data),
            'video_track_used': video_track_index,
            'video_info': video_info,
            'ffmpeg_used': ffmpeg_process is not None,
            'opencv_fallback_used': ffmpeg_process is None,
            'frames_processed': frames_processed,
            'processing_successful': frames_processed > 0
        }
        
        logger.info(f"Video processing completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error in video processing: {e}")
        raise