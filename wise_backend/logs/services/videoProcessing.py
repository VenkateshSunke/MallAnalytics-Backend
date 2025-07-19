import os
import time
import logging
import cv2
import numpy as np
import ffmpeg
import gc
import psutil
from datetime import datetime
from .awsRecognitionService import AWSRekognitionService
from .personTracker import PersonTracker
from .movementLogger import MovementLogger
from .blueprint_utils import get_blueprint_mapping_by_camera_id

logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        logger.warning("psutil not available, memory monitoring disabled")
        return 0
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")
        return 0

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()

def check_memory_limit(current_usage_mb, limit_mb):
    """Check if memory usage exceeds limit"""
    return current_usage_mb > limit_mb

def validate_processing_parameters(batch_size, memory_limit_mb):
    """Validate processing parameters"""
    if batch_size < 1:
        logger.warning(f"Invalid batch_size {batch_size}, using default 10")
        batch_size = 10
    
    if memory_limit_mb < 100:
        logger.warning(f"Memory limit too low {memory_limit_mb} MB, using minimum 100 MB")
        memory_limit_mb = 100
    elif memory_limit_mb > 8192:
        logger.warning(f"Memory limit too high {memory_limit_mb} MB, using maximum 8192 MB")
        memory_limit_mb = 8192
    
    return batch_size, memory_limit_mb

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

def start_process(camera, output_path, track_index=None, batch_size=10, memory_limit_mb=1024):
    """
    Process video for analytics and movement tracking using ffmpeg-python with enhanced memory management
    
    Args:
        camera: Camera configuration object from cameras.py
        output_path: Path to the video file to process
        track_index: Optional track index to select (e.g., 0 for track1, 1 for track2, etc.)
        batch_size: Number of frames to process in memory before cleanup (default: 10)
        memory_limit_mb: Memory limit in MB for frame processing (default: 1024)
    
    Returns:
        dict: Processing results and statistics
    """
    process = None
    output_process = None
    frame_buffer = []  # Buffer for batch processing
    
    try:
        logger.info(f"Starting video processing for camera: {camera}")
        logger.info(f"Video path: {output_path}")
        
        # Validate input parameters first
        if not camera:
            raise ValueError("Camera configuration is required")
        
        if not output_path:
            raise ValueError("Output path is required")
        
        # Validate processing parameters
        batch_size, memory_limit_mb = validate_processing_parameters(batch_size, memory_limit_mb)
        logger.info(f"Processing parameters: batch_size={batch_size}, memory_limit={memory_limit_mb} MB")
        
        # Use output_path as the video path (since that's what we're processing)
        video_path = output_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check if file is readable
        if not os.access(video_path, os.R_OK):
            raise PermissionError(f"Cannot read video file: {video_path}")
        
        file_size = os.path.getsize(video_path)
        logger.info(f"Video file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError(f"Video file is empty: {video_path}")
        
        # Check for suspiciously small files (likely corrupted)
        if file_size < 10000:  # Less than 10KB
            logger.warning(f"Video file is very small ({file_size} bytes) - may be corrupted: {video_path}")
            raise ValueError(f"Video file appears to be corrupted or too small ({file_size} bytes): {video_path}")
        
        logger.info(f"Video file validation passed: {file_size} bytes")
        
        # Initialize services with better error handling
        try:
            aws_service = AWSRekognitionService()
        except Exception as e:
            logger.warning(f"Failed to initialize AWS service: {e}")
            aws_service = None
        
        try:
            person_tracker = PersonTracker(camera_id=str(camera.get('id', 'unknown')))
        except Exception as e:
            logger.error(f"Failed to initialize person tracker: {e}")
            raise
        
        try:
            movement_logger = MovementLogger()
        except Exception as e:
            logger.error(f"Failed to initialize movement logger: {e}")
            raise
        
        # Setup AWS if available
        aws_enabled = False
        if aws_service:
            try:
                success, message = aws_service.enable_aws_rekognition()
                if success:
                    aws_service.set_export_mode(True)
                    aws_enabled = True
                    logger.info("AWS Rekognition enabled successfully")
                else:
                    logger.warning(f"AWS Rekognition not enabled: {message}")
            except Exception as e:
                logger.warning(f"Error enabling AWS: {e}")
        
        # Probe video with better error handling
        logger.info("Probing video file...")
        try:
            probe = ffmpeg.probe(video_path)
            logger.info("Video probe successful")
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg probe failed: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode() if e.stderr else 'No stderr'}")
            
            # Try with error tolerance
            try:
                logger.info("Retrying probe with error tolerance...")
                probe = ffmpeg.probe(video_path, v='error', select_streams='v:0')
                logger.info("Video probe successful with error tolerance")
            except Exception as retry_e:
                logger.error(f"FFmpeg probe failed even with error tolerance: {retry_e}")
                raise ValueError(f"Could not probe video file: {video_path}")
        except Exception as e:
            logger.error(f"Unexpected error during video probe: {e}")
            raise
        
        # Extract video stream information
        video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        
        if not video_streams:
            raise ValueError("No video streams found in the file")
        
        logger.info(f"Found {len(video_streams)} video stream(s)")
        
        # Select video stream
        if track_index is not None:
            if track_index >= len(video_streams):
                raise ValueError(f"Invalid track_index {track_index}, only {len(video_streams)} tracks available.")
            selected_stream = video_streams[track_index]
            logger.info(f"Selected video track {track_index} (stream index {selected_stream['index']})")
        else:
            selected_stream = video_streams[0]
            logger.info(f"Using default video track (stream index {selected_stream['index']})")
        
        # Store the original video streams info for logging
        original_video_streams = video_streams.copy()
        
        # Get video properties from selected stream
        try:
            width = int(selected_stream['width'])
            height = int(selected_stream['height'])
        except (KeyError, ValueError) as e:
            logger.error(f"Could not get video dimensions: {e}")
            raise ValueError("Invalid video dimensions")
        
        # Get FPS with better error handling
        try:
            fps_str = selected_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                fps_parts = fps_str.split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1])
            else:
                fps = float(fps_str)
            
            if fps <= 0 or fps > 1000:  # Sanity check
                logger.warning(f"Invalid FPS value: {fps}, using default 30")
                fps = 30.0
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Could not parse FPS, using default: {e}")
            fps = 30.0
        
        # Get total frames
        total_frames = int(selected_stream.get('nb_frames', 0))
        if total_frames == 0:
            try:
                duration = float(probe['format'].get('duration', 0))
                total_frames = int(duration * fps) if duration > 0 else 0
            except (ValueError, TypeError):
                total_frames = 0
        
        logger.info(f"Video properties: {width}x{height} @ {fps:.2f}fps")
        if total_frames > 0:
            logger.info(f"Estimated total frames: {total_frames}")
        else:
            logger.warning("Could not determine total frame count")
        
        # Setup FFmpeg input process with better error handling
        logger.info("Setting up FFmpeg input process...")
        try:
            # Build FFmpeg input command
            input_args = {'loglevel': 'error'}  # Reduce FFmpeg log noise
            
            # Add error tolerance for corrupted files
            input_args.update({
                'err_detect': 'ignore_err',  # Ignore errors and continue
                'fflags': '+genpts+discardcorrupt',  # Generate timestamps and discard corrupt frames
                'max_error_rate': '0.0'  # Allow some errors
            })
            
            # Add additional parameters for severely corrupted files
            input_args.update({
                'analyzeduration': '1000000',  # Analyze longer duration
                'probesize': '1000000'  # Larger probe size
            })
            
            # Handle track selection more robustly
            if track_index is not None and track_index < len(video_streams):
                # Use map to select specific video stream
                selected_stream = video_streams[track_index]
                stream_index = selected_stream['index']
                # Use stream selection in the output, not input
                input_stream = ffmpeg.input(video_path, **input_args)
                logger.info(f"Using video track {track_index} (stream index {stream_index})")
            else:
                # Use default input - let FFmpeg choose the first video stream
                input_stream = ffmpeg.input(video_path, **input_args)
                logger.info(f"Using default video track (no specific track selection)")
            
            # Setup output to pipe
            if track_index is not None and track_index < len(video_streams):
                # Apply stream selection in the output
                selected_stream = video_streams[track_index]
                stream_index = selected_stream['index']
                
                # Use the same logic as the working implementation
                output_stream = input_stream.output(
                    'pipe:', 
                    format='rawvideo', 
                    pix_fmt='bgr24',
                    loglevel='error',
                    map=f'0:{stream_index}'  # Use stream index directly like working implementation
                )
                logger.info(f"Selected video stream 0:{stream_index} (track {track_index})")
            else:
                # Use default video stream
                output_stream = input_stream.video.output(
                    'pipe:', 
                    format='rawvideo', 
                    pix_fmt='bgr24',
                    loglevel='error'
                )
            
            # Start the process
            process = output_stream.run_async(pipe_stdout=True, pipe_stderr=True)
            
            # Check if process started successfully
            time.sleep(0.5)  # Give process more time to start
            if process.poll() is not None:
                # Process failed to start
                stderr_output = ""
                if process.stderr:
                    try:
                        stderr_output = process.stderr.read().decode()
                    except:
                        stderr_output = "Could not read stderr"
                
                logger.error(f"FFmpeg input process failed to start. Return code: {process.poll()}")
                logger.error(f"FFmpeg stderr: {stderr_output}")
                raise RuntimeError(f"FFmpeg input process failed: {stderr_output}")
            
            logger.info("FFmpeg input process started successfully")
            
        except Exception as e:
            logger.error(f"Error setting up FFmpeg input process: {e}")
            if process:
                try:
                    process.terminate()
                except:
                    pass
            raise
        
        # Setup output video path - handle different video formats
        base_name = os.path.splitext(output_path)[0]
        extension = os.path.splitext(output_path)[1]
        processed_output_path = f"{base_name}_processed{extension}"
        
        output_dir = os.path.dirname(processed_output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                logger.error(f"Failed to create output directory: {e}")
                raise
        
        # Check if output directory is writable
        if output_dir:
            if not os.access(output_dir, os.W_OK):
                logger.error(f"Output directory is not writable: {output_dir}")
                raise PermissionError(f"Cannot write to output directory: {output_dir}")
            logger.info(f"Output directory is writable: {output_dir}")
        else:
            # No directory specified, use current directory
            if not os.access('.', os.W_OK):
                logger.error("Current directory is not writable")
                raise PermissionError("Cannot write to current directory")
            logger.info("Using current directory for output")
        
        # Setup FFmpeg output process
        logger.info("Setting up FFmpeg output process...")
        try:
            output_process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
                .output(
                    processed_output_path, 
                    vcodec='libx264', 
                    pix_fmt='yuv420p', 
                    r=fps,
                    loglevel='error'
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=True)
            )
            
            # Check if output process started
            time.sleep(0.1)
            if output_process.poll() is not None:
                stderr_output = ""
                if output_process.stderr:
                    try:
                        stderr_output = output_process.stderr.read().decode()
                    except:
                        stderr_output = "Could not read stderr"
                
                logger.error(f"FFmpeg output process failed to start. Return code: {output_process.poll()}")
                logger.error(f"FFmpeg stderr: {stderr_output}")
                raise RuntimeError(f"FFmpeg output process failed: {stderr_output}")
            
            logger.info(f"FFmpeg output process started successfully. Output: {processed_output_path}")
            
        except Exception as e:
            logger.error(f"Error setting up FFmpeg output process: {e}")
            if output_process:
                try:
                    output_process.terminate()
                except:
                    pass
            raise
        
        # Get blueprint mapping data
        camera_id = str(camera.get('id', 'unknown'))
        logger.info(f"Loading blueprint data for camera: {camera_id}")
        
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
            logger.warning(f"No blueprint mapping found for camera {camera_id}, using camera config stores")
            stores = camera.get('stores', {})
            calibration_data = None
        
        # Initialize processing variables
        frame_count = 0
        aws_frame_skip = 60  # Process AWS every 60th frame
        processing_start_time = time.time()
        store_entry_logged = {}  # Track store entries
        
        # Performance tracking
        timings = {
            'aws_detection': 0.0,
            'yolo_detection': 0.0,
            'tracking_update': 0.0,
            'movement_logging': 0.0,
            'frame_processing': 0.0,
            'memory_management': 0.0
        }
        
        tracked_people = {}
        frame_size = width * height * 3  # BGR format
        
        # Memory management variables
        initial_memory = get_memory_usage()
        memory_check_interval = 50  # Check memory every 50 frames
        last_memory_check = 0
        
        logger.info(f"Starting frame processing. Frame size: {frame_size} bytes")
        logger.info(f"AWS enabled: {aws_enabled}, Frame skip: {aws_frame_skip}")
        logger.info(f"Video properties: {width}x{height} @ {fps:.2f}fps")
        logger.info(f"Selected video stream: {selected_stream.get('index', 'unknown')}")
        logger.info(f"Available video streams: {len(original_video_streams)}")
        for i, stream in enumerate(original_video_streams):
            logger.info(f"  Stream {i}: index={stream.get('index')}, codec={stream.get('codec_name')}, "
                       f"resolution={stream.get('width')}x{stream.get('height')}")
        
        # Process frames
        consecutive_read_failures = 0
        max_read_failures = 20  # Increased from 10 to be more tolerant
        consecutive_processing_errors = 0
        max_processing_errors = 10
        
        logger.info("Starting frame reading loop...")
        logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
        
        while True:
            frame_start_time = time.time()
            
            # Memory management check
            if frame_count - last_memory_check >= memory_check_interval:
                current_memory = get_memory_usage()
                memory_increase = current_memory - initial_memory
                logger.debug(f"Memory usage: {current_memory:.1f} MB (increase: {memory_increase:.1f} MB)")
                
                if check_memory_limit(current_memory, memory_limit_mb):
                    logger.warning(f"Memory limit exceeded ({current_memory:.1f} MB > {memory_limit_mb} MB). Forcing cleanup.")
                    cleanup_memory()
                    timings['memory_management'] += time.time() - frame_start_time
                
                last_memory_check = frame_count
            
            try:
                # Read frame data from FFmpeg
                logger.debug(f"Attempting to read frame {frame_count + 1}")
                in_bytes = process.stdout.read(frame_size)
                
                if not in_bytes:
                    logger.info(f"No more data from FFmpeg (EOF) at frame {frame_count}")
                    # Check if process is still running
                    if process.poll() is not None:
                        return_code = process.poll()
                        logger.info(f"FFmpeg process ended with return code: {return_code}")
                        # Read any stderr output
                        if process.stderr:
                            try:
                                stderr_output = process.stderr.read().decode()
                                if stderr_output:
                                    logger.error(f"FFmpeg stderr: {stderr_output}")
                            except Exception as e:
                                logger.warning(f"Could not read FFmpeg stderr: {e}")
                    else:
                        logger.warning("FFmpeg process still running but no data received")
                    break
                
                if len(in_bytes) < frame_size:
                    consecutive_read_failures += 1
                    logger.warning(f"Incomplete frame read: got {len(in_bytes)} bytes, expected {frame_size}. Failure count: {consecutive_read_failures}")
                    
                    if consecutive_read_failures >= max_read_failures:
                        logger.error(f"Too many consecutive read failures ({consecutive_read_failures}). Stopping processing.")
                        break
                    
                    # Try to recover by reading more data
                    remaining_bytes = frame_size - len(in_bytes)
                    additional_bytes = process.stdout.read(remaining_bytes)
                    if additional_bytes:
                        in_bytes += additional_bytes
                        if len(in_bytes) == frame_size:
                            consecutive_read_failures = 0  # Reset on successful recovery
                            logger.info("Successfully recovered incomplete frame read")
                        else:
                            continue  # Still incomplete, skip this frame
                    else:
                        continue  # No additional data, skip this frame
                
                # Reset failure counter on successful read
                consecutive_read_failures = 0
                
                # Convert bytes to numpy array
                try:
                    frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                except ValueError as e:
                    logger.error(f"Failed to reshape frame data: {e}")
                    continue
                
                current_time = frame_count / fps
                face_detections = []
                
                # AWS face detection (every N frames)
                if aws_enabled and aws_service and frame_count % aws_frame_skip == 0:
                    try:
                        t_aws_start = time.time()
                        face_detections = aws_service.detect_faces(frame, current_time)
                        timings['aws_detection'] += time.time() - t_aws_start
                        
                        if face_detections:
                            logger.info(f"Frame {frame_count}: Found {len(face_detections)} registered faces")
                    except Exception as e:
                        logger.warning(f"AWS face detection failed on frame {frame_count}: {e}")
                        face_detections = []
                
                # YOLO person detection
                try:
                    t_yolo_start = time.time()
                    detections = person_tracker.analyze_frame(frame)
                    timings['yolo_detection'] += time.time() - t_yolo_start
                except Exception as e:
                    logger.warning(f"YOLO detection failed on frame {frame_count}: {e}")
                    detections = {'persons': []}
                
                # Update tracking
                try:
                    t_update_start = time.time()
                    tracked_people = person_tracker.update(
                        detections.get('persons', []), stores, frame_count, face_detections, frame
                    )
                    timings['tracking_update'] += time.time() - t_update_start
                except Exception as e:
                    logger.warning(f"Tracking update failed on frame {frame_count}: {e}")
                    tracked_people = {}
                
                # Draw annotations
                try:
                    frame_with_annotations = frame.copy()
                    
                    # Draw store polygons first
                    frame_with_annotations = draw_store_polygons(frame_with_annotations, stores, calibration_data)
                    
                    # Draw face detections
                    for face in face_detections:
                        bbox = face.get('bbox', [0, 0, 0, 0])
                        if len(bbox) >= 4:
                            x, y, w, h = bbox[:4]
                            confidence = face.get('confidence', 0)
                            user_id = face.get('user_id', 'Unknown')
                            
                            # Draw face bounding box (red)
                            cv2.rectangle(frame_with_annotations, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            
                            # Draw label
                            label = f"{user_id} ({confidence:.1f}%)"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(frame_with_annotations, (x, y - label_size[1] - 10), 
                                         (x + label_size[0], y), (0, 0, 255), -1)
                            cv2.putText(frame_with_annotations, label, (x, y - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Draw person tracking
                    for person_id, person in tracked_people.items():
                        try:
                            bbox = person.get('bbox', [0, 0, 0, 0])
                            if len(bbox) >= 4:
                                x, y, w, h = bbox[:4]
                                location = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
                                
                                # Get person identity
                                user_id, face_id = person_tracker.get_person_identity(person_id)
                                activity_type = 'walking' if person.get('is_moving', False) else 'standing'
                                
                                # Check store entry
                                current_store = person.get('current_store')
                                if current_store and current_store in stores:
                                    current_timestamp = datetime.now()
                                    should_log_entry = (
                                        person_id not in store_entry_logged or 
                                        current_store not in store_entry_logged[person_id] or
                                        (current_timestamp - store_entry_logged[person_id][current_store]).seconds > 5
                                    )
                                    
                                    if should_log_entry:
                                        store_name = stores[current_store].get('name', current_store)
                                        logger.info(f"Person {person_id} entered store: {store_name}")
                                        
                                        if person_id not in store_entry_logged:
                                            store_entry_logged[person_id] = {}
                                        store_entry_logged[person_id][current_store] = current_timestamp
                                        
                                        # Log movement for registered persons
                                        if user_id:
                                            try:
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
                                            except Exception as e:
                                                logger.warning(f"Failed to log store entry: {e}")
                                
                                # Draw person bounding box (blue)
                                cv2.rectangle(frame_with_annotations, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                
                                # Draw person label
                                display_name = person_tracker.get_person_display_name(person_id)
                                person_label = f"{display_name} ({activity_type})"
                                if current_store and current_store in stores:
                                    store_name = stores[current_store].get('name', current_store)
                                    person_label += f" in {store_name}"
                                
                                label_size = cv2.getTextSize(person_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                cv2.rectangle(frame_with_annotations, (x, y + h), 
                                             (x + label_size[0], y + h + label_size[1] + 10), (255, 0, 0), -1)
                                cv2.putText(frame_with_annotations, person_label, (x, y + h + label_size[1] + 5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                                # Log general movement
                                if user_id:
                                    try:
                                        movement_logger.log_person_movement(
                                            person_id=user_id,
                                            location=location,
                                            timestamp=datetime.now(),
                                            camera_id=camera_id,
                                            confidence=person.get('confidence', 1.0),
                                            store_id=current_store,
                                            activity_type=activity_type,
                                            bbox=bbox,
                                            face_id=face_id
                                        )
                                    except Exception as e:
                                        logger.debug(f"Failed to log movement: {e}")
                        
                        except Exception as e:
                            logger.warning(f"Error processing person {person_id}: {e}")
                            continue
                
                except Exception as e:
                    logger.warning(f"Error drawing annotations on frame {frame_count}: {e}")
                    frame_with_annotations = frame  # Use original frame
                
                # Write frame to output
                try:
                    output_process.stdin.write(frame_with_annotations.tobytes())
                    logger.debug(f"Successfully wrote frame {frame_count} to output")
                except Exception as e:
                    logger.error(f"Failed to write frame {frame_count} to output: {e}")
                    logger.error(f"Output process status: {output_process.poll()}")
                    break
                
                # Check if processes are still running with enhanced error handling
                if process.poll() is not None:
                    return_code = process.poll()
                    stderr_output = ""
                    if process.stderr:
                        try:
                            stderr_output = process.stderr.read().decode()
                        except:
                            stderr_output = "Could not read stderr"
                    
                    if return_code == 0:
                        logger.info("FFmpeg input process completed successfully")
                    else:
                        logger.error(f"FFmpeg input process terminated with return code: {return_code}")
                        if stderr_output:
                            logger.error(f"FFmpeg input stderr: {stderr_output}")
                    
                    # Check for specific error conditions
                    if return_code == -9:  # SIGKILL
                        logger.error("FFmpeg process was killed due to memory issues")
                    elif return_code == -11:  # SIGSEGV
                        logger.error("FFmpeg process crashed due to segmentation fault")
                    elif return_code == -6:  # SIGABRT
                        logger.error("FFmpeg process aborted")
                    
                    break
                
                if output_process.poll() is not None:
                    return_code = output_process.poll()
                    stderr_output = ""
                    if output_process.stderr:
                        try:
                            stderr_output = output_process.stderr.read().decode()
                        except:
                            stderr_output = "Could not read stderr"
                    
                    if return_code == 0:
                        logger.info("FFmpeg output process completed successfully")
                    else:
                        logger.error(f"FFmpeg output process terminated with return code: {return_code}")
                        if stderr_output:
                            logger.error(f"FFmpeg output stderr: {stderr_output}")
                    break
                
                # Update counters and timing
                frame_count += 1
                timings['frame_processing'] += time.time() - frame_start_time
                
                # Batch processing and memory cleanup
                if frame_count % batch_size == 0:
                    # Clear frame buffer and force garbage collection
                    frame_buffer.clear()
                    cleanup_memory()
                    
                    # Log memory usage
                    current_memory = get_memory_usage()
                    logger.debug(f"Batch cleanup completed. Memory usage: {current_memory:.1f} MB")
                
                # Progress logging with memory information
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - processing_start_time
                    fps_current = frame_count / elapsed_time if elapsed_time > 0 else 0
                    current_memory = get_memory_usage()
                    
                    if total_frames > 0:
                        progress = (frame_count / total_frames) * 100
                        eta_seconds = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
                        eta_min = int(eta_seconds // 60)
                        eta_sec = int(eta_seconds % 60)
                        logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                                  f"Speed: {fps_current:.1f} fps - ETA: {eta_min:02d}:{eta_sec:02d} - "
                                  f"Memory: {current_memory:.1f} MB")
                    else:
                        logger.info(f"Processed {frame_count} frames - Speed: {fps_current:.1f} fps - "
                                  f"Memory: {current_memory:.1f} MB")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                consecutive_processing_errors += 1
                if consecutive_processing_errors >= max_processing_errors:
                    logger.error(f"Too many processing errors. Stopping.")
                    break
                continue
        
        logger.info(f"Frame processing completed. Total frames processed: {frame_count}")
        
    except Exception as e:
        logger.error(f"Critical error in video processing: {e}")
        raise
        
    finally:
        # Cleanup processes
        logger.info("Cleaning up FFmpeg processes...")
        
        if process:
            try:
                if process.stdout:
                    process.stdout.close()
                process.terminate()
                process.wait(timeout=5)
                logger.info("Input process cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up input process: {e}")
                try:
                    process.kill()
                except:
                    pass
        
        if output_process:
            try:
                if output_process.stdin:
                    output_process.stdin.close()
                    logger.debug("Output process stdin closed")
                output_process.terminate()
                output_process.wait(timeout=5)
                logger.info("Output process cleaned up")
                
                # Check if output file was created
                if os.path.exists(processed_output_path):
                    file_size = os.path.getsize(processed_output_path)
                    logger.info(f"Output video created: {processed_output_path} ({file_size} bytes)")
                else:
                    logger.warning(f"Output video file not found: {processed_output_path}")
                    
            except Exception as e:
                logger.warning(f"Error cleaning up output process: {e}")
                try:
                    output_process.kill()
                except:
                    pass
    
    # Calculate final results
    processing_time = time.time() - processing_start_time
    aws_calls = aws_service.api_calls_count if aws_service else 0
    final_memory = get_memory_usage()
    memory_peak = final_memory - initial_memory if 'initial_memory' in locals() else 0
    
    # Final memory cleanup
    cleanup_memory()
    
    results = {
        'total_frames': total_frames if total_frames > 0 else frame_count,
        'processed_frames': frame_count,
        'processing_time': processing_time,
        'processing_speed': frame_count / processing_time if processing_time > 0 else 0,
        'aws_api_calls': aws_calls,
        'tracked_persons': len(tracked_people),
        'input_video_path': video_path,
        'processed_video_path': processed_output_path,
        'performance_timings': timings,
        'aws_frame_skip': aws_frame_skip,
        'camera_id': camera_id,
        'blueprint_used': bool(blueprint_mapping),
        'stores_processed': len(stores),
        'store_entries_logged': len(store_entry_logged),
        'calibration_available': bool(calibration_data and 'store_matrices' in calibration_data),
        'selected_track_index': track_index,
        'available_video_tracks': len(original_video_streams) if 'original_video_streams' in locals() else 0,
        'memory_usage': {
            'initial_mb': initial_memory if 'initial_memory' in locals() else 0,
            'final_mb': final_memory,
            'peak_increase_mb': memory_peak,
            'batch_size': batch_size,
            'memory_limit_mb': memory_limit_mb
        }
    }
    
    logger.info(f"Video processing completed successfully: {results}")
    
    # Log final memory statistics
    if results['memory_usage']['final_mb'] > 0:
        logger.info(f"Final memory usage: {results['memory_usage']['final_mb']:.1f} MB "
                   f"(peak increase: {results['memory_usage']['peak_increase_mb']:.1f} MB)")
    
    # Log performance summary
    logger.info(f"Performance summary: {frame_count} frames processed in {processing_time:.1f}s "
               f"({results['processing_speed']:.1f} fps)")
    
    return results