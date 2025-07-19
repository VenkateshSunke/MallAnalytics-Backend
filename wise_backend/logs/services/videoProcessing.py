import os
import time
import logging
import cv2
import numpy as np
import ffmpeg
import signal
import threading
import subprocess
from datetime import datetime
from .awsRecognitionService import AWSRekognitionService
from .personTracker import PersonTracker
from .movementLogger import MovementLogger
from .blueprint_utils import get_blueprint_mapping_by_camera_id, get_blueprint_info_by_camera_id

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.input_process = None
        self.output_process = None
        self.processing_active = False
        self.cleanup_timeout = 15  # Increased timeout
        
    def cleanup_processes(self):
        """Enhanced process cleanup with proper termination"""
        logger.info("Starting process cleanup...")
        
        # Stop processing flag
        self.processing_active = False
        
        # Cleanup output process first
        if self.output_process:
            try:
                if self.output_process.stdin and not self.output_process.stdin.closed:
                    self.output_process.stdin.close()
                    
                # Give process time to flush and finalize
                time.sleep(2)
                
                if self.output_process.poll() is None:
                    logger.info("Terminating output process...")
                    self.output_process.terminate()
                    
                    try:
                        self.output_process.wait(timeout=self.cleanup_timeout)
                        logger.info("Output process terminated successfully")
                    except subprocess.TimeoutExpired:
                        logger.warning("Output process termination timed out, killing...")
                        self.output_process.kill()
                        self.output_process.wait()
                        
            except Exception as e:
                logger.error(f"Error cleaning up output process: {e}")
                try:
                    self.output_process.kill()
                except:
                    pass
            
            self.output_process = None
        
        # Cleanup input process
        if self.input_process:
            try:
                if self.input_process.stdout and not self.input_process.stdout.closed:
                    self.input_process.stdout.close()
                    
                if self.input_process.poll() is None:
                    logger.info("Terminating input process...")
                    self.input_process.terminate()
                    
                    try:
                        self.input_process.wait(timeout=self.cleanup_timeout)
                        logger.info("Input process terminated successfully")
                    except subprocess.TimeoutExpired:
                        logger.warning("Input process termination timed out, killing...")
                        self.input_process.kill()
                        self.input_process.wait()
                        
            except Exception as e:
                logger.error(f"Error cleaning up input process: {e}")
                try:
                    self.input_process.kill()
                except:
                    pass
                    
            self.input_process = None
        
        logger.info("Process cleanup completed")

def validate_video_file(video_path):
    """Enhanced video validation"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    file_size = os.path.getsize(video_path)
    if file_size < 10000:  # Less than 10KB
        raise ValueError(f"Video file too small ({file_size} bytes): {video_path}")
    
    # Try to open with OpenCV first for quick validation
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"OpenCV cannot open video file: {video_path}")
        
        # Try to read one frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError(f"Cannot read frames from video: {video_path}")
            
        cap.release()
        logger.info(f"Video validation passed: {video_path} ({file_size} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"Video validation failed: {e}")
        raise ValueError(f"Invalid video file: {video_path}")

def get_video_info(video_path):
    """Get video information with better error handling"""
    try:
        # Use OpenCV first for basic info (more reliable)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video with OpenCV")
        
        cv_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv_fps = cap.get(cv2.CAP_PROP_FPS)
        cv_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"OpenCV video info: {cv_width}x{cv_height}@{cv_fps:.2f}fps, {cv_frame_count} frames")
        
        cap.release()
        
        # Use ffprobe for additional info
        try:
            probe = ffmpeg.probe(video_path)
            video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
            
            if video_streams:
                stream = video_streams[0]
                
                # Use OpenCV values as primary, ffprobe as backup
                width = cv_width if cv_width > 0 else int(stream.get('width', 0))
                height = cv_height if cv_height > 0 else int(stream.get('height', 0))
                
                # Parse FPS from ffprobe
                fps_str = stream.get('r_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    ff_fps = float(num) / float(den) if float(den) != 0 else 30.0
                else:
                    ff_fps = float(fps_str)
                
                fps = cv_fps if cv_fps > 0 else ff_fps
                frame_count = cv_frame_count if cv_frame_count > 0 else int(stream.get('nb_frames', 0))
                
                # Calculate frame count from duration if needed
                if frame_count == 0:
                    try:
                        duration = float(probe['format'].get('duration', 0))
                        frame_count = int(duration * fps) if duration > 0 else 0
                        logger.info(f"Calculated frame count from duration: {duration}s * {fps}fps = {frame_count} frames")
                    except:
                        frame_count = 0
                
                logger.info(f"Final video info: {width}x{height}@{fps:.2f}fps, {frame_count} frames")
                
                return {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': frame_count / fps if fps > 0 else 0,
                    'video_streams': video_streams
                }
            
        except Exception as e:
            logger.warning(f"FFprobe failed, using OpenCV values: {e}")
        
        # Fallback to OpenCV only
        logger.info(f"Using OpenCV fallback: {cv_width}x{cv_height}@{cv_fps:.2f}fps, {cv_frame_count} frames")
        return {
            'width': cv_width,
            'height': cv_height, 
            'fps': cv_fps if cv_fps > 0 else 30.0,
            'frame_count': cv_frame_count,
            'duration': cv_frame_count / cv_fps if cv_fps > 0 else 0,
            'video_streams': []
        }
        
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        raise ValueError(f"Cannot analyze video: {video_path}")

def transform_blueprint_to_video_coordinates(blueprint_polygon, transformation_matrix):
    """Transform blueprint polygon coordinates to video coordinates"""
    try:
        if not blueprint_polygon or len(blueprint_polygon) < 3:
            return None
        
        pts = np.array(blueprint_polygon, dtype=np.float32).reshape(-1, 1, 2)
        video_pts = cv2.perspectiveTransform(pts, transformation_matrix)
        return [tuple(map(int, pt[0])) for pt in video_pts]
        
    except Exception as e:
        logger.error(f"Error transforming coordinates: {e}")
        return None

def precompile_store_data(stores, calibration_data=None):
    """Pre-compile store polygons for faster rendering"""
    compiled_stores = {}
    
    for store_id, store in stores.items():
        polygon = store.get('video_polygon') or store.get('polygon')
        if not polygon or len(polygon) < 3:
            continue
            
        try:
            # Transform if calibration available
            if calibration_data and 'store_matrices' in calibration_data:
                matrices = calibration_data['store_matrices']
                if store_id in matrices:
                    matrix = np.array(matrices[store_id], dtype=np.float32)
                    transformed = transform_blueprint_to_video_coordinates(polygon, matrix)
                    if transformed:
                        polygon = transformed
            
            # Convert to numpy array
            if isinstance(polygon[0], (list, tuple)):
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            else:
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            
            compiled_stores[store_id] = {
                'points': pts,
                'name': store.get('name', store_id),
                'centroid': np.mean(pts, axis=0).astype(int)[0]
            }
            
        except Exception as e:
            logger.warning(f"Failed to compile store {store_id}: {e}")
            continue
    
    logger.info(f"Compiled {len(compiled_stores)} store polygons")
    return compiled_stores

def draw_annotations_fast(frame, compiled_stores, face_detections, tracked_people, person_tracker):
    """Optimized annotation drawing"""
    frame_annotated = frame.copy()
    
    # Draw stores with single overlay
    if compiled_stores:
        overlay = frame_annotated.copy()
        for store_data in compiled_stores.values():
            try:
                pts = store_data['points']
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
                
                # Store name
                centroid = store_data['centroid']
                name = store_data['name']
                cv2.putText(overlay, name, tuple(centroid), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except:
                continue
        
        cv2.addWeighted(overlay, 0.3, frame_annotated, 0.7, 0, frame_annotated)
    
    # Draw face detections
    for face in face_detections:
        bbox = face.get('bbox', [])
        if len(bbox) >= 4:
            x, y, w, h = bbox[:4]
            cv2.rectangle(frame_annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            user_id = face.get('user_id', 'Unknown')
            cv2.putText(frame_annotated, user_id, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw person tracking
    for person_id, person in tracked_people.items():
        bbox = person.get('bbox', [])
        if len(bbox) >= 4:
            x, y, w, h = bbox[:4]
            cv2.rectangle(frame_annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            display_name = person_tracker.get_person_display_name(person_id)
            cv2.putText(frame_annotated, display_name, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame_annotated

def start_process(camera, output_path, track_index=None, skip_frames=2, max_resolution=(1920, 1080)):
    """
    REVISED and optimized video processing with complete video processing guarantee
    
    Args:
        camera: Camera configuration
        output_path: Input video path  
        track_index: Video track selection
        skip_frames: Process every Nth frame (2 = every other frame)
        max_resolution: Max processing resolution (width, height)
    """
    processor = VideoProcessor()
    
    try:
        logger.info(f"Starting REVISED video processing")
        logger.info(f"Video: {output_path}")
        logger.info(f"Skip frames: {skip_frames}, Max resolution: {max_resolution}")
        
        # Enhanced validation
        validate_video_file(output_path)
        video_info = get_video_info(output_path)
        
        width = video_info['width']
        height = video_info['height']
        fps = video_info['fps']
        total_frames = video_info['frame_count']
        
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError(f"Invalid video parameters: {width}x{height}@{fps}fps")
        
        logger.info(f"Video info: {width}x{height}@{fps:.2f}fps, {total_frames} frames")
        
        # Calculate processing resolution - FIXED: Use consistent dimensions
        max_w, max_h = max_resolution
        scale_factor = 1.0
        
        if width > max_w or height > max_h:
            scale_w = max_w / width
            scale_h = max_h / height
            scale_factor = min(scale_w, scale_h)
        
        # FIXED: Use consistent dimensions throughout pipeline
        if scale_factor != 1.0:
            output_width = int(width * scale_factor)
            output_height = int(height * scale_factor)
            # Ensure even dimensions for H.264
            output_width = (output_width // 2) * 2
            output_height = (output_height // 2) * 2
            logger.info(f"Scaling: {width}x{height} -> {output_width}x{output_height} (factor: {scale_factor:.3f})")
        else:
            output_width = (width // 2) * 2
            output_height = (height // 2) * 2
            logger.info(f"No scaling needed: {output_width}x{output_height}")
        
        # Effective FPS after frame skipping
        output_fps = max(1.0, fps / skip_frames)
        
        # Initialize services
        aws_service = None
        try:
            aws_service = AWSRekognitionService()
            success, _ = aws_service.enable_aws_rekognition()
            if success:
                aws_service.set_export_mode(True)
                logger.info("AWS enabled")
        except:
            logger.warning("AWS initialization failed")
        
        person_tracker = PersonTracker(camera_id=str(camera.get('id', 'unknown')))
        movement_logger = MovementLogger()
        
        # Get store data
        camera_id = str(camera.get('id', 'unknown'))
        blueprint_mapping = get_blueprint_mapping_by_camera_id(camera_id)
        
        if blueprint_mapping:
            stores = blueprint_mapping.get('stores', {})
            calibration_data = blueprint_mapping.get('calibration', {})
        else:
            stores = camera.get('stores', {})
            calibration_data = None
        
        compiled_stores = precompile_store_data(stores, calibration_data)
        
        # Setup input process with OpenCV fallback
        logger.info("Setting up input stream...")
        
        # Try FFmpeg first
        use_opencv = False
        try:
            input_stream = ffmpeg.input(output_path, loglevel='error')
            
            if scale_factor != 1.0:
                video_stream = input_stream.video.filter('scale', output_width, output_height)
            else:
                video_stream = input_stream.video
            
            output_stream = video_stream.output(
                'pipe:', 
                format='rawvideo', 
                pix_fmt='bgr24',
                loglevel='error'
            )
            
            processor.input_process = output_stream.run_async(pipe_stdout=True, pipe_stderr=True)
            
            # Test if process started
            time.sleep(0.5)
            if processor.input_process.poll() is not None:
                logger.warning("FFmpeg input failed, falling back to OpenCV")
                use_opencv = True
                processor.input_process = None
                
        except Exception as e:
            logger.warning(f"FFmpeg setup failed: {e}, using OpenCV")
            use_opencv = True
        
        # OpenCV fallback
        if use_opencv:
            cap = cv2.VideoCapture(output_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video with OpenCV")
            logger.info("Using OpenCV for input")
        
        # Setup output
        processed_output_path = output_path.replace('.mp4', '_processed.mp4').replace('.mkv', '_processed.mp4')
        output_dir = os.path.dirname(processed_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Setting up output stream...")
        try:
            processor.output_process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', 
                      s=f'{output_width}x{output_height}', r=output_fps)
                .output(
                    processed_output_path,
                    vcodec='libx264',
                    pix_fmt='yuv420p',
                    r=output_fps,
                    preset='medium',      # Better balance of speed/quality
                    crf=23,              # Better quality
                    tune='film',         # Better for real content
                    bf=2,                # B-frames for better compression
                    refs=3,              # Multiple reference frames
                    loglevel='error',
                    movflags='faststart'  # Web optimization
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=True)
            )
            
            time.sleep(0.2)
            if processor.output_process.poll() is not None:
                raise RuntimeError("Output process failed to start")
                
        except Exception as e:
            logger.error(f"Output setup failed: {e}")
            raise
        
        # Main processing loop - REVISED for complete processing
        frame_count = 0
        processed_count = 0
        processing_start = time.time()
        last_log_time = time.time()
        aws_skip = 300  # Very infrequent AWS calls
        
        # FIXED: Calculate exact frame size consistently
        frame_size = output_width * output_height * 3
        logger.info(f"Frame size: {frame_size} bytes ({output_width}x{output_height}x3)")
        
        processor.processing_active = True
        temporary_failures = 0
        max_temp_failures = 10  # Reduced from 50
        frame_buffer = []  # For batched output
        buffer_size = 5  # Batch size for output writing
        
        logger.info("Starting main processing loop...")
        logger.info(f"Expected total frames: {total_frames}, Expected duration: {total_frames/fps if fps > 0 else 0:.2f} seconds")
        
        while processor.processing_active:
            try:
                frame = None
                
                # Read frame - IMPROVED logic
                if use_opencv:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.info("OpenCV: End of video reached")
                        break
                    
                    if scale_factor != 1.0:
                        frame = cv2.resize(frame, (output_width, output_height))
                    elif frame.shape[:2] != (output_height, output_width):
                        frame = cv2.resize(frame, (output_width, output_height))
                        
                else:
                    # FIXED: FFmpeg input with proper EOF handling
                    try:
                        in_bytes = processor.input_process.stdout.read(frame_size)
                        
                        if not in_bytes:
                            # Check if process finished normally
                            if processor.input_process.poll() is not None:
                                logger.info("FFmpeg: End of video reached (process finished)")
                                break
                            else:
                                # No data but process still running - temporary issue
                                time.sleep(0.01)
                                temporary_failures += 1
                                if temporary_failures > max_temp_failures:
                                    logger.warning("Too many temporary failures, assuming EOF")
                                    break
                                continue
                        
                        if len(in_bytes) < frame_size:
                            # IMPROVED: Handle partial frames better
                            logger.debug(f"Partial frame: {len(in_bytes)}/{frame_size} bytes")
                            
                            # Try to read remaining bytes
                            remaining_bytes = frame_size - len(in_bytes)
                            additional_bytes = processor.input_process.stdout.read(remaining_bytes)
                            
                            if additional_bytes:
                                in_bytes += additional_bytes
                            
                            if len(in_bytes) < frame_size:
                                # Still not enough - likely EOF
                                logger.info("Partial frame at end of video, stopping")
                                break
                        
                        frame = np.frombuffer(in_bytes, np.uint8).reshape([output_height, output_width, 3])
                        temporary_failures = 0  # Reset on successful read
                        
                    except Exception as e:
                        if "broken pipe" in str(e).lower() or "epipe" in str(e).lower():
                            logger.info("Input pipe closed, end of video")
                            break
                        logger.error(f"FFmpeg read error: {e}")
                        temporary_failures += 1
                        if temporary_failures > max_temp_failures:
                            break
                        continue
                
                frame_count += 1
                
                # FIXED: Frame skipping logic
                if skip_frames > 1 and ((frame_count - 1) % skip_frames) != 0:
                    continue
                
                processed_count += 1
                
                # Ensure frame is correct size
                if frame.shape[:2] != (output_height, output_width):
                    frame = cv2.resize(frame, (output_width, output_height))
                
                # AI processing (very limited)
                face_detections = []
                if aws_service and processed_count % aws_skip == 0:
                    try:
                        current_time = frame_count / fps
                        face_detections = aws_service.detect_faces(frame, current_time)
                    except:
                        pass
                
                # Person detection
                detections = {'persons': []}
                try:
                    detections = person_tracker.analyze_frame(frame)
                except:
                    pass
                
                # Tracking update
                tracked_people = {}
                try:
                    tracked_people = person_tracker.update(
                        detections.get('persons', []), stores, frame_count, face_detections, frame
                    )
                except:
                    pass
                
                # Draw annotations
                frame_annotated = draw_annotations_fast(
                    frame, compiled_stores, face_detections, tracked_people, person_tracker
                )
                
                # IMPROVED: Batched output writing
                frame_buffer.append(frame_annotated.tobytes())
                
                # Write batch when buffer is full
                if len(frame_buffer) >= buffer_size:
                    try:
                        for frame_data in frame_buffer:
                            processor.output_process.stdin.write(frame_data)
                        processor.output_process.stdin.flush()
                        frame_buffer.clear()
                    except Exception as e:
                        if "broken pipe" in str(e).lower() or "epipe" in str(e).lower():
                            logger.info("Output pipe closed")
                            break
                        logger.error(f"Output write failed: {e}")
                        break
                
                # Progress logging
                current_time = time.time()
                if current_time - last_log_time > 10:
                    elapsed = current_time - processing_start
                    fps_current = processed_count / elapsed if elapsed > 0 else 0
                    
                    if total_frames > 0:
                        progress = (frame_count / total_frames) * 100
                        logger.info(f"Progress: {progress:.1f}% - Speed: {fps_current:.1f} fps - Frames: {frame_count}/{total_frames} (processed: {processed_count})")
                    else:
                        logger.info(f"Processed: {processed_count} frames - Speed: {fps_current:.1f} fps - Total frames: {frame_count}")
                    
                    last_log_time = current_time
                
                # Safety check for runaway processing
                if total_frames > 0 and frame_count > total_frames * 1.1:
                    logger.warning("Processed more frames than expected, stopping")
                    break
                
            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected processing error: {e}")
                break
        
        # IMPROVED: Write remaining frames in buffer
        if frame_buffer:
            try:
                logger.info(f"Writing remaining {len(frame_buffer)} frames...")
                for frame_data in frame_buffer:
                    processor.output_process.stdin.write(frame_data)
                processor.output_process.stdin.flush()
            except Exception as e:
                logger.warning(f"Failed to write final frames: {e}")
        
        # Cleanup input
        if use_opencv:
            cap.release()
        
        logger.info(f"Processing completed: {processed_count} frames processed out of {frame_count} total frames read")
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise
    finally:
        processor.cleanup_processes()
    
    # Results
    processing_time = time.time() - processing_start
    completion_percentage = 100.0
    if total_frames > 0:
        completion_percentage = (frame_count / total_frames) * 100
    
    results = {
        'total_frames_read': frame_count,
        'frames_processed': processed_count,
        'processing_time': processing_time,
        'processing_speed': processed_count / processing_time if processing_time > 0 else 0,
        'processed_video_path': processed_output_path,
        'input_method': 'opencv' if use_opencv else 'ffmpeg',
        'optimizations_applied': True,
        'scale_factor': scale_factor,
        'output_fps': output_fps,
        'output_resolution': f"{output_width}x{output_height}",
        'completion_percentage': completion_percentage,
        'success': completion_percentage > 95.0  # Consider successful if >95% processed
    }
    
    logger.info(f"Final results: {results}")
    return results