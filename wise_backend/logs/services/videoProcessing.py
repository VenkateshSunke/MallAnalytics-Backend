import os
import time
import logging
import cv2
import numpy as np
import subprocess
import json
import threading
import queue
from datetime import datetime
from .awsRecognitionService import AWSRekognitionService
from .personTracker import PersonTracker
from .movementLogger import MovementLogger
from .blueprint_utils import get_blueprint_mapping_by_camera_id, get_blueprint_info_by_camera_id

logger = logging.getLogger(__name__)

class FrameReader:
    """Robust frame reader with buffering and error recovery"""
    
    def __init__(self, ffmpeg_process, frame_size, buffer_size=10):
        self.ffmpeg_process = ffmpeg_process
        self.frame_size = frame_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.reading_thread = None
        self.stop_event = threading.Event()
        self.error_count = 0
        self.max_errors = 10
        
    def start_reading(self):
        """Start the frame reading thread"""
        self.reading_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.reading_thread.start()
        
    def _read_frames(self):
        """Thread function to read frames from FFmpeg"""
        while not self.stop_event.is_set():
            try:
                # Read frame data with timeout handling
                raw_frame = self._read_frame_with_timeout()
                
                if raw_frame is None:
                    logger.info("FrameReader: FFmpeg stream ended or failed, stopping frame reader thread")
                    break
                    
                if len(raw_frame) == self.frame_size:
                    # Convert to numpy array
                    frame = np.frombuffer(raw_frame, dtype=np.uint8)
                    try:
                        self.frame_queue.put(frame, timeout=1.0)
                        self.error_count = 0  # Reset error count on success
                    except queue.Full:
                        logger.warning("Frame queue full, dropping frame")
                        continue
                else:
                    logger.warning(f"Incomplete frame: expected {self.frame_size}, got {len(raw_frame)}")
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        logger.error("Too many frame read errors, stopping")
                        break
                        
            except Exception as e:
                logger.error(f"Error in frame reading thread: {e}")
                self.error_count += 1
                if self.error_count >= self.max_errors:
                    break
                time.sleep(0.1)  # Brief pause on error
                
    def _read_frame_with_timeout(self, timeout=5.0):
        """Read a complete frame with timeout"""
        try:
            # Check if process is still running
            if self.ffmpeg_process.poll() is not None:
                logger.info("FrameReader: FFmpeg process already exited")
                return None
                
            # Read frame data in chunks to handle partial reads
            remaining = self.frame_size
            frame_data = b''
            start_time = time.time()
            
            while remaining > 0 and (time.time() - start_time) < timeout:
                # Read in chunks of up to 65536 bytes (common buffer size)
                chunk_size = min(remaining, 65536)
                try:
                    chunk = self.ffmpeg_process.stdout.read(chunk_size)
                    if not chunk:  # EOF
                        break
                    frame_data += chunk
                    remaining -= len(chunk)
                except Exception as e:
                    logger.warning(f"Error reading chunk: {e}")
                    break
                    
            return frame_data if len(frame_data) == self.frame_size else None
            
        except Exception as e:
            logger.error(f"Error in _read_frame_with_timeout: {e}")
            return None
            
    def get_frame(self, timeout=1.0):
        """Get next frame from queue"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def stop(self):
        """Stop the frame reader"""
        self.stop_event.set()
        if self.reading_thread and self.reading_thread.is_alive():
            self.reading_thread.join(timeout=2.0)

def get_video_info_robust(video_path):
    """Get video information with enhanced error handling"""
    try:
        # First, try to get basic info
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', '-show_format', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(result.stdout)
        
        # Find the best video stream
        video_streams = [s for s in data.get('streams', []) if s.get('codec_type') == 'video']
        if not video_streams:
            raise ValueError("No video streams found")
            
        # Filter out thumbnail/attachment streams
        main_streams = []
        for stream in video_streams:
            disposition = stream.get('disposition', {})
            if disposition.get('attached_pic') == 1:
                continue
            width = int(stream.get('width', 0))
            height = int(stream.get('height', 0))
            if width < 100 or height < 100:
                continue
            main_streams.append(stream)
            
        if not main_streams:
            main_streams = video_streams  # Fallback to any video stream
            
        # Use the first suitable stream
        video_stream = main_streams[0]
        stream_index = video_streams.index(video_stream)
        
        # Extract properties
        width = int(video_stream.get('width', 640))
        height = int(video_stream.get('height', 480))
        
        # Get FPS
        fps_str = video_stream.get('r_frame_rate', '30/1')
        try:
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(fps_str)
        except (ValueError, ZeroDivisionError):
            fps = 30.0
            
        # Get duration and calculate total frames
        duration = 0
        if 'duration' in video_stream:
            duration = float(video_stream['duration'])
        elif 'format' in data and 'duration' in data['format']:
            duration = float(data['format']['duration'])
            
        total_frames = int(duration * fps) if duration > 0 else 0
        
        # If no frame count, try to get it directly
        if 'nb_frames' in video_stream and video_stream['nb_frames']:
            try:
                direct_frames = int(video_stream['nb_frames'])
                if direct_frames > 0:
                    total_frames = direct_frames
            except ValueError:
                pass
                
        logger.info(f"Video info: {width}x{height} @ {fps:.2f}fps, duration: {duration:.2f}s, frames: {total_frames}")
        return fps, width, height, total_frames, stream_index
        
    except subprocess.TimeoutExpired:
        logger.error("FFprobe timed out")
        return 30.0, 640, 480, 0, 0
    except subprocess.CalledProcessError as e:
        logger.error(f"FFprobe failed: {e}")
        return 30.0, 640, 480, 0, 0
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return 30.0, 640, 480, 0, 0

def create_ffmpeg_reader(video_path, stream_index=0):
    """Create FFmpeg reader process with optimal settings"""
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-map', f'0:v:{stream_index}',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-vsync', '0',  # Don't duplicate or drop frames
        '-threads', '2',  # Limit threads to prevent resource issues
        '-'
    ]
    
    logger.info(f"Starting FFmpeg reader: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Unbuffered for immediate reading
            preexec_fn=None  # Don't modify process attributes
        )
        
        # Test if process started successfully
        time.sleep(0.5)
        if process.poll() is not None:
            stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg failed to start: {stderr_output}")
            
        return process
        
    except Exception as e:
        logger.error(f"Failed to create FFmpeg reader: {e}")
        raise

def create_ffmpeg_writer(output_path, width, height, fps):
    """Create FFmpeg writer process with optimal settings"""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'medium',  # Balance between speed and quality
        '-crf', '23',  # Good quality
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',  # Optimize for streaming
        '-threads', '2',
        output_path
    ]
    
    logger.info(f"Starting FFmpeg writer: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        return process
        
    except Exception as e:
        logger.error(f"Failed to create FFmpeg writer: {e}")
        raise

def transform_blueprint_to_video_coordinates(blueprint_polygon, transformation_matrix):
    """Transform blueprint polygon coordinates to video coordinates using perspective transformation"""
    try:
        if not blueprint_polygon or len(blueprint_polygon) < 3:
            return None
        
        pts = np.array(blueprint_polygon, dtype=np.float32).reshape(-1, 1, 2)
        video_pts = cv2.perspectiveTransform(pts, transformation_matrix)
        video_polygon = [tuple(map(int, pt[0])) for pt in video_pts]
        return video_polygon
        
    except Exception as e:
        logger.error(f"Error transforming blueprint coordinates: {e}")
        return None

def draw_store_polygons(frame, stores, calibration_data=None):
    """Draw store polygons on the frame"""
    frame_with_stores = frame.copy()
    
    for store_id, store in stores.items():
        polygon = store.get('video_polygon') or store.get('polygon')
        
        if not polygon or len(polygon) < 3:
            continue
            
        try:
            # Transform coordinates if calibration data is available
            if calibration_data and 'store_matrices' in calibration_data:
                store_matrices = calibration_data['store_matrices']
                if store_id in store_matrices:
                    matrix = np.array(store_matrices[store_id], dtype=np.float32)
                    video_polygon = transform_blueprint_to_video_coordinates(polygon, matrix)
                    if video_polygon:
                        polygon = video_polygon
                    else:
                        continue
                else:
                    continue
            
            # Convert to numpy array for drawing
            if isinstance(polygon[0], (list, tuple)):
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            else:
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            
            # Draw filled semi-transparent polygon
            overlay = frame_with_stores.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, frame_with_stores, 0.7, 0, frame_with_stores)
            
            # Draw polygon outline
            cv2.polylines(frame_with_stores, [pts], True, (0, 255, 0), 2)
            
            # Draw store name at centroid
            centroid = np.mean(pts, axis=0).astype(int)[0]
            store_name = store.get('name', store_id)
            
            text_size = cv2.getTextSize(store_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame_with_stores, 
                         (centroid[0] - text_size[0]//2 - 5, centroid[1] - text_size[1] - 5),
                         (centroid[0] + text_size[0]//2 + 5, centroid[1] + 5),
                         (0, 0, 0), -1)
            
            cv2.putText(frame_with_stores, store_name, 
                       (centroid[0] - text_size[0]//2, centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"Error drawing polygon for store {store_id}: {e}")
            continue
    
    return frame_with_stores

def start_process(camera, output_path, stream_override=None):
    """Process video for analytics and movement tracking with improved error handling"""
    frame_reader = None
    ffmpeg_reader = None
    ffmpeg_writer = None
    
    try:
        logger.info(f"Starting video processing for camera: {camera}")
        
        # Initialize services
        aws_service = AWSRekognitionService()
        person_tracker = PersonTracker(camera_id=str(camera.get('id', 'unknown')))
        movement_logger = MovementLogger()
        
        # Enable AWS for testing
        aws_enabled = True
        success, message = aws_service.enable_aws_rekognition()
        if not success:
            logger.warning(f"AWS Rekognition not enabled: {message}")
        aws_service.set_export_mode(True)
        
        # Validate video path
        video_path = output_path
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video properties
        fps, width, height, total_frames, stream_index = get_video_info_robust(video_path)
        frame_size = height * width * 3

        if stream_override is not None:
            logger.info(f"Overriding video stream index to {stream_override}")
            stream_index = stream_override
        
        # Create FFmpeg processes
        ffmpeg_reader = create_ffmpeg_reader(video_path, stream_index)
        
        processed_output_path = output_path.replace('.mp4', '_processed.mp4').replace('.mkv', '_processed.mp4')
        ffmpeg_writer = create_ffmpeg_writer(processed_output_path, width, height, fps)
        
        # Create robust frame reader
        frame_reader = FrameReader(ffmpeg_reader, frame_size, buffer_size=30)
        frame_reader.start_reading()
        
        # Get blueprint mapping
        camera_id = str(camera.get('id', 'unknown'))
        blueprint_mapping = get_blueprint_mapping_by_camera_id(camera_id)
        
        if blueprint_mapping:
            stores = blueprint_mapping.get('stores', {})
            calibration_data = blueprint_mapping.get('calibration', {})
        else:
            stores = camera.get('stores', {})
            calibration_data = None
        
        # Processing variables
        frame_count = 0
        aws_frame_skip = 60
        processing_start_time = time.time()
        store_entry_logged = {}
        tracked_people = {}
        
        timings = {
            'aws_detection': 0.0,
            'yolo_detection': 0.0,
            'tracking_update': 0.0,
            'movement_logging': 0.0
        }
        
        logger.info(f"Starting frame processing. Frame size: {frame_size} bytes")
        
        # Process frames
        consecutive_timeouts = 0
        max_consecutive_timeouts = 10
        
        while consecutive_timeouts < max_consecutive_timeouts:
            try:
                # Get frame from reader
                raw_frame = frame_reader.get_frame(timeout=2.0)
                
                if raw_frame is None:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        logger.info("No more frames available, ending processing")
                        break
                    continue
                
                consecutive_timeouts = 0  # Reset timeout counter
                
                # Reshape frame
                frame = raw_frame.reshape((height, width, 3))
                current_time = frame_count / fps
                
                # AWS face detection
                face_detections = []
                if frame_count % aws_frame_skip == 0 and aws_service.aws_enabled:
                    t_aws_start = time.time()
                    face_detections = aws_service.detect_faces(frame, current_time)
                    timings['aws_detection'] += time.time() - t_aws_start
                    if face_detections:
                        logger.info(f"Frame {frame_count}: Found {len(face_detections)} registered faces")
                
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
                
                # Create annotated frame
                frame_with_annotations = frame.copy()
                frame_with_annotations = draw_store_polygons(frame_with_annotations, stores, calibration_data)
                
                # Draw face detections
                for face in face_detections:
                    bbox = face['bbox']
                    x, y, w, h = bbox
                    confidence = face.get('confidence', 0)
                    user_id = face.get('user_id', 'Unknown')
                    
                    cv2.rectangle(frame_with_annotations, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    label = f"{user_id} ({confidence:.1f}%)"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame_with_annotations, (x, y - label_size[1] - 10), 
                                 (x + label_size[0], y), (0, 0, 255), -1)
                    cv2.putText(frame_with_annotations, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw person tracking and handle store entries
                for person_id, person in tracked_people.items():
                    bbox = person['bbox']
                    x, y, w, h = bbox
                    location = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
                    
                    user_id, face_id = person_tracker.get_person_identity(person_id)
                    activity_type = 'walking' if person.get('is_moving', False) else 'standing'
                    
                    # Handle store entry
                    current_store = person.get('current_store')
                    if current_store and current_store in stores:
                        current_timestamp = datetime.now()
                        if (person_id not in store_entry_logged or 
                            current_store not in store_entry_logged[person_id] or
                            (current_timestamp - store_entry_logged[person_id][current_store]).seconds > 5):
                            
                            store_name = stores[current_store].get('name', current_store)
                            logger.info(f"Person {person_id} entered store: {store_name}")
                            
                            if person_id not in store_entry_logged:
                                store_entry_logged[person_id] = {}
                            
                            store_entry_logged[person_id][current_store] = current_timestamp
                            
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
                    
                    # Draw person bounding box
                    cv2.rectangle(frame_with_annotations, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
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
                    
                    # Log movement
                    if user_id:
                        movement_logger.log_person_movement(
                            person_id=user_id,
                            location=location,
                            timestamp=datetime.now(),
                            camera_id=camera_id,
                            confidence=person.get('confidence', 1.0),
                            store_id=person.get('current_store'),
                            activity_type=activity_type,
                            bbox=bbox,
                            face_id=face_id
                        )
                
                # Write frame
                try:
                    ffmpeg_writer.stdin.write(frame_with_annotations.tobytes())
                    ffmpeg_writer.stdin.flush()
                except BrokenPipeError:
                    logger.error("FFmpeg writer pipe broken, stopping processing")
                    break
                except Exception as write_error:
                    logger.error(f"Error writing frame {frame_count}: {write_error}")
                    break
                
                frame_count += 1
                
                # Progress logging
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                    
            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                frame_count += 1
                continue
        
        # Calculate results
        processing_time = time.time() - processing_start_time
        results = {
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'processing_time': processing_time,
            'processing_speed': frame_count / processing_time if processing_time > 0 else 0,
            'aws_api_calls': aws_service.api_calls_count if hasattr(aws_service, 'api_calls_count') else 0,
            'tracked_persons': len(tracked_people),
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
        logger.error(f"Error in video processing: {e}")
        logger.exception("Full exception traceback:")
        raise
    finally:
        # Cleanup
        try:
            if frame_reader:
                frame_reader.stop()
            if ffmpeg_reader:
                try:
                    if ffmpeg_reader.stdout:
                        ffmpeg_reader.stdout.close()
                    if ffmpeg_reader.stderr:
                        ffmpeg_reader.stderr.close()
                    if ffmpeg_reader.poll() is None:
                        ffmpeg_reader.terminate()
                        ffmpeg_reader.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg reader did not terminate in time, killing it")
                    ffmpeg_reader.kill()
                except Exception as e:
                    logger.error(f"Exception during FFmpeg reader cleanup: {e}")
            if ffmpeg_writer:
                if ffmpeg_writer.stdin:
                    ffmpeg_writer.stdin.close()
                if ffmpeg_writer.poll() is None:
                    ffmpeg_writer.terminate()
                    ffmpeg_writer.wait(timeout=5)
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")