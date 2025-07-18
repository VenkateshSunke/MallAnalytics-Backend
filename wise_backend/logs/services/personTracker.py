import logging
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class PersonTracker:
    """Person tracking and movement detection without GUI dependencies"""
    
    def __init__(self, camera_id="cam001"):
        # Load YOLO model - handle PyTorch 2.6 compatibility issues
        self.model = None
        
        # Try different model loading approaches
        model_options = [
            'yolov8n',  # Download fresh model (preferred)
            'yolov8s',  # Alternative model
            'yolov8m',  # Another alternative
        ]
        
        for model_name in model_options:
            try:
                logger.info(f"Attempting to load YOLO model: {model_name}")
                # Force fresh download to avoid PyTorch 2.6 issues
                self.model = YOLO(model_name, task='detect')
                logger.info(f"YOLO model loaded successfully: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if self.model is None:
            logger.error("Failed to load any YOLO model - video processing will be limited")
        else:
            logger.info("YOLO model initialization completed successfully")
        self.camera_id = camera_id
        
        # Tracking parameters
        self.next_id = 1
        self.tracked_people = {}  # id -> {bbox, last_seen, current_store, history, confidence, timestamp, is_moving, position_history, name, face_id}
        self.store_entry_threshold = 0.5
        self.max_frames_missing = 10
        self.iou_threshold = 0.15
        self.min_confidence = 0.25
        self.max_movement = 150
        self.velocity_smoothing = 0.5
        self.last_positions = {}
        self.track_history = {}
        self.max_history = 10
        self.face_detection_history = {}
        self.max_face_history = 2
        self.last_store = {}
        self.frame_counter = 0

        # YOLO class indices
        self.PERSON_CLASS = 0
        
        # Processing control
        self.last_results = {
            'faces': [], 
            'persons': [],
            'objects': {}
        }
        
        # Movement detection parameters
        self.motion_threshold = 2.0
        self.position_history_size = 4
        self.idle_threshold = 15
        self.min_movement_threshold = 0.5
        self.max_movement_threshold = 200.0
        self.velocity_threshold = 0.3
        self.movement_history_size = 3
        self.movement_persistence = 2
        self.idle_persistence = 3
        self.min_consecutive_movement_frames = 1
        self.jitter_threshold = 0.5
        self.confidence_movement_factor = 0.3
        self.movement_scale_factor = 1.5
        self.min_person_height = 50
        self.max_person_height = 400
        
        # Movement state tracking
        self.movement_state = {}
        
        # Face-Person association parameters (enhanced like BlueprintTrack)
        self.face_person_iou_threshold = 0.01  # Extremely loose matching
        self.face_person_distance_threshold = 100  # Max distance between face and person centers
        self.name_persistence_frames = 10  # How many frames to keep a name association
        self.face_confidence_threshold = 50.0  # Lowered threshold for testing
        
        # CCTV-optimized movement thresholds (like BlueprintTrack)
        self.motion_threshold = 2.0  # Decreased for CCTV's lower frame rate
        self.position_history_size = 4  # Reduced for faster response in CCTV
        self.idle_threshold = 15  # Reduced for CCTV's lower frame rate
        self.min_movement_threshold = 0.5  # Much lower for overhead CCTV perspective
        self.max_movement_threshold = 200.0  # Increased to handle faster movements
        self.velocity_threshold = 0.3  # Much lower for overhead CCTV perspective
        self.movement_history_size = 3  # Reduced for faster response
        self.movement_persistence = 2  # Reduced for faster response
        self.idle_persistence = 3  # Reduced for faster response
        self.min_consecutive_movement_frames = 1  # Single frame movement detection
        self.jitter_threshold = 0.5  # Lower to detect smaller movements in overhead view
        self.confidence_movement_factor = 0.3  # More lenient confidence requirements for overhead
        self.movement_scale_factor = 1.5  # Scale movement based on person size
        self.min_person_height = 50  # Minimum height to consider for movement
        self.max_person_height = 400  # Maximum height to consider for movement
        
        # Name tracking for persons
        self.person_names = {}  # person_id -> {'name': str, 'confidence': float, 'last_seen': frame, 'face_id': str}
        self.name_assignment_history = {}
        
        # Debugging flags
        self.debug_mode = False
        self.debug_movement = False

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

    def calculate_velocity(self, old_bbox, new_bbox):
        """Calculate velocity between two bounding boxes"""
        if old_bbox is None or new_bbox is None:
            return (0, 0)
        
        old_center_x = old_bbox[0] + old_bbox[2] / 2
        old_center_y = old_bbox[1] + old_bbox[3] / 2
        new_center_x = new_bbox[0] + new_bbox[2] / 2
        new_center_y = new_bbox[1] + new_bbox[3] / 2
        
        return (new_center_x - old_center_x, new_center_y - old_center_y)

    def calculate_velocity_magnitude(self, velocity):
        """Calculate magnitude of velocity vector"""
        return np.sqrt(velocity[0]**2 + velocity[1]**2)

    def is_person_in_store(self, person_bbox, store_polygon):
        """Check if person is inside a store polygon"""
        if not store_polygon or len(store_polygon) < 3:
            return False
        
        # Get person center
        person_center_x = person_bbox[0] + person_bbox[2] / 2
        person_center_y = person_bbox[1] + person_bbox[3] / 2
        
        # Create polygon and check if point is inside
        try:
            # Ensure store_polygon is a list of (x, y) tuples
            if isinstance(store_polygon, str):
                # If it's a JSON string, parse it
                import json
                store_polygon = json.loads(store_polygon)
            
            # Convert to list of tuples if it's a flat list
            if len(store_polygon) % 2 == 0 and len(store_polygon) >= 6:
                # Flat list of coordinates [x1, y1, x2, y2, ...]
                polygon_points = [(float(store_polygon[i]), float(store_polygon[i+1])) for i in range(0, len(store_polygon), 2)]
            else:
                # Already in correct format or invalid
                polygon_points = [(float(p[0]), float(p[1])) for p in store_polygon]
            
            # Ensure we have at least 3 points for a valid polygon
            if len(polygon_points) < 3:
                return False
            
            # Create a simple point-in-polygon check without Shapely to avoid geometry issues
            return self._point_in_polygon(person_center_x, person_center_y, polygon_points)
        except Exception as e:
            logger.error(f"Error checking store intersection: {e}")
            return False
    
    def _point_in_polygon(self, x, y, polygon):
        """Simple point-in-polygon test using ray casting algorithm"""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def cleanup_old_tracks(self, current_frame):
        """Remove old tracks that haven't been seen recently"""
        current_time = current_frame / 30.0  # Assuming 30 FPS
        to_remove = []
        
        for person_id, person in self.tracked_people.items():
            if current_time - person['last_seen'] > self.max_frames_missing / 30.0:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.tracked_people[person_id]
            if person_id in self.last_positions:
                del self.last_positions[person_id]

    def find_best_match(self, detection, unmatched_tracks):
        """Find best matching track for a detection"""
        best_match = None
        best_iou = 0
        
        for person_id in unmatched_tracks:
            person = self.tracked_people[person_id]
            iou = self.calculate_iou(detection, person['bbox'])
            
            if iou > best_iou and iou > self.iou_threshold:
                best_iou = iou
                best_match = person_id
        
        return best_match

    def update_store_status(self, person_id, person, stores, current_time, frame_number):
        """Update store status for a person"""
        old_store = person.get('current_store')
        current_store = None
        
        # Check which store the person is in
        for store_id, store in stores.items():
            # Try to get video_polygon first (transformed coordinates), then fall back to polygon
            store_polygon = store.get('video_polygon') or store.get('polygon')
            if store_polygon and self.is_person_in_store(person['bbox'], store_polygon):
                current_store = store_id
                break
        
        person['current_store'] = current_store
        
        # Log store entry/exit events
        if old_store != current_store:
            if old_store is not None:
                logger.info(f"Person {person_id} exited store {old_store}")
            if current_store is not None:
                logger.info(f"Person {person_id} entered store {current_store}")

    def create_new_track(self, detection, frame_number, current_time, stores):
        """Create a new track for a detection"""
        person_id = f"person_{self.next_id}"
        self.next_id += 1
        
        # Determine current store
        current_store = None
        for store_id, store in stores.items():
            # Try to get video_polygon first (transformed coordinates), then fall back to polygon
            store_polygon = store.get('video_polygon') or store.get('polygon')
            if store_polygon and self.is_person_in_store(detection, store_polygon):
                current_store = store_id
                break
        
        # Initialize movement state
        self.movement_state[person_id] = {
            'is_moving': False,
            'movement_frames': 0,
            'idle_frames': 0,
            'last_movement_time': current_time,
            'position_history': []
        }
        
        self.tracked_people[person_id] = {
            'bbox': detection,
            'last_seen': current_time,
            'current_store': current_store,
            'history': [detection],
            'confidence': 1.0,
            'timestamp': current_time,
            'is_moving': False,
            'position_history': [detection],
            'name': None,
            'face_id': None
        }
        
        return person_id

    def update_movement_status(self, person_id, person, old_bbox, new_bbox, velocity):
        """Update movement status for a person"""
        if person_id not in self.movement_state:
            self.movement_state[person_id] = {
                'is_moving': False,
                'movement_frames': 0,
                'idle_frames': 0,
                'last_movement_time': person['timestamp'],
                'position_history': []
            }
        
        movement_state = self.movement_state[person_id]
        velocity_magnitude = self.calculate_velocity_magnitude(velocity)
        
        # Update position history
        movement_state['position_history'].append(new_bbox)
        if len(movement_state['position_history']) > self.position_history_size:
            movement_state['position_history'].pop(0)
        
        # Determine if person is moving
        is_moving = velocity_magnitude > self.motion_threshold
        
        if is_moving:
            movement_state['movement_frames'] += 1
            movement_state['idle_frames'] = 0
            movement_state['last_movement_time'] = person['timestamp']
        else:
            movement_state['idle_frames'] += 1
            movement_state['movement_frames'] = 0
        
        # Update movement state
        if movement_state['movement_frames'] >= self.movement_persistence:
            movement_state['is_moving'] = True
        elif movement_state['idle_frames'] >= self.idle_persistence:
            movement_state['is_moving'] = False
        
        # Update person's movement status
        person['is_moving'] = movement_state['is_moving']

    def analyze_frame(self, frame):
        """Analyze frame using YOLO for person detection"""
        if self.model is None:
            logger.warning("YOLO model not loaded, skipping frame analysis")
            return {'persons': []}
            
        try:
            results = self.model(frame, verbose=False)
            
            persons = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if it's a person (class 0)
                        if int(box.cls[0]) == self.PERSON_CLASS:
                            confidence = float(box.conf[0])
                            if confidence > self.min_confidence:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                                persons.append(bbox)
            
            return {'persons': persons}
        except Exception as e:
            logger.error(f"Error in YOLO analysis: {e}")
            return {'persons': []}

    def update(self, detected_people, stores, frame_number, face_detections=None, current_frame=None):
        """Update tracking with new detections"""
        current_time = frame_number / 30.0  # Assuming 30 FPS
        
        # Clean up old tracks
        self.cleanup_old_tracks(frame_number)
        
        # Associate faces with persons if face detections are available
        if face_detections:
            self.associate_faces_with_persons(face_detections, frame_number)
        
        # Match detections to existing tracks
        matched_detections = set()
        unmatched_tracks = set(self.tracked_people.keys())
        
        for detection in detected_people:
            best_match = self.find_best_match(detection, unmatched_tracks)
            if best_match:
                # Update existing track
                person = self.tracked_people[best_match]
                old_bbox = person['bbox']
                person['bbox'] = detection
                person['last_seen'] = current_time
                person['timestamp'] = current_time
                
                # Update history
                person['history'].append(detection)
                if len(person['history']) > self.max_history:
                    person['history'].pop(0)
                
                # Calculate velocity and update movement
                velocity = self.calculate_velocity(old_bbox, detection)
                self.update_movement_status(best_match, person, old_bbox, detection, velocity)
                
                # Update store status
                self.update_store_status(best_match, person, stores, current_time, frame_number)
                
                matched_detections.add(best_match)
                unmatched_tracks.remove(best_match)
            else:
                # Create new track
                self.create_new_track(detection, frame_number, current_time, stores)
        
        # Handle unmatched tracks (people no longer detected)
        for person_id in unmatched_tracks:
            person = self.tracked_people[person_id]
            person['last_seen'] = current_time
        
        # Clean up old face detections
        self.cleanup_old_face_detections(frame_number)
        
        return self.tracked_people

    def associate_faces_with_persons(self, face_detections, current_frame):
        """Associate detected faces with tracked persons based on IoU"""
        if not face_detections:
            return
        
        # Filter for registered faces only
        registered_faces = [f for f in face_detections if f.get('detection_type') == 'registered_face']
        
        # Initialize face detections for all persons
        for person_id, person in self.tracked_people.items():
            if 'face_detections' not in person:
                person['face_detections'] = []
        
        # Associate faces with persons based on IoU
        for face in registered_faces:
            face_bbox = face['bbox']
            best_person_id = None
            best_iou = 0
            
            for person_id, person in self.tracked_people.items():
                person_bbox = person['bbox']
                iou = self.calculate_iou(face_bbox, person_bbox)
                
                if iou > best_iou and iou > self.face_person_iou_threshold:
                    best_iou = iou
                    best_person_id = person_id
            
            if best_person_id:
                person = self.tracked_people[best_person_id]
                
                # Add face detection with relative position
                face_with_relative_pos = {
                    'bbox': face_bbox,
                    'user_id': face.get('user_id'),
                    'face_id': face.get('face_id'),
                    'confidence': face.get('confidence', 0),
                    'timestamp': face.get('timestamp', current_frame / 30.0),
                    'frame_number': current_frame
                }
                
                person['face_detections'].append(face_with_relative_pos)
                
                # Update person's face_id and name if this is a better match
                if face.get('confidence', 0) > self.face_confidence_threshold:
                    person['face_id'] = face.get('face_id')
                    person['name'] = face.get('user_id')
                    
                    # Update name assignment history
                    self.name_assignment_history[best_person_id] = {
                        'name': face.get('user_id'),
                        'confidence': face.get('confidence', 0),
                        'last_seen': current_frame,
                        'face_id': face.get('face_id')
                    }
                
                logger.info(f"Associated face {face.get('user_id', 'Unknown')} with person {best_person_id} (IoU: {best_iou:.3f})")

    def cleanup_old_face_detections(self, current_frame):
        """Clean up old face detections to prevent memory bloat"""
        current_time = current_frame / 30.0
        max_face_age = 2.0  # Keep face detections for 2 seconds
        
        for person_id, person in self.tracked_people.items():
            if 'face_detections' in person and person['face_detections']:
                valid_face_detections = []
                
                for face_detection in person['face_detections']:
                    face_time = face_detection.get('timestamp', face_detection.get('frame_number', 0) / 30.0)
                    if current_time - face_time <= max_face_age:
                        valid_face_detections.append(face_detection)
                
                person['face_detections'] = valid_face_detections
                
                # Clear name if no recent face detections
                if not valid_face_detections and person_id in self.name_assignment_history:
                    last_assignment = self.name_assignment_history[person_id]
                    if current_frame - last_assignment['last_seen'] > self.name_persistence_frames:
                        person['name'] = None
                        person['face_id'] = None

    def calculate_face_person_spatial_relationship(self, face_bbox, person_bbox):
        """Calculate spatial relationship between face and person bounding boxes"""
        # Convert face bbox to same format as person bbox if needed
        if len(face_bbox) == 4:
            fx, fy, fw, fh = face_bbox
        else:
            # Handle different bbox formats
            fx, fy, fw, fh = face_bbox
        
        px, py, pw, ph = person_bbox
        
        # Calculate IOU
        x1 = max(fx, px)
        y1 = max(fy, py)
        x2 = min(fx + fw, px + pw)
        y2 = min(fy + fh, py + ph)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, False
        
        intersection = (x2 - x1) * (y2 - y1)
        face_area = fw * fh
        person_area = pw * ph
        union = face_area + person_area - intersection
        
        iou = intersection / union if union > 0 else 0
        
        # Also calculate if face is contained within person
        face_center_x = fx + fw / 2
        face_center_y = fy + fh / 2
        
        is_face_in_person = (px <= face_center_x <= px + pw and 
                            py <= face_center_y <= py + ph)
        
        return iou, is_face_in_person

    def calculate_distance_between_centers(self, bbox1, bbox2):
        """Calculate distance between centers of two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2
        center2_x = x2 + w2 / 2
        center2_y = y2 + h2 / 2
        
        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

    def get_person_identity(self, person_id):
        """Get the best available identity for a person"""
        person = self.tracked_people.get(person_id)
        if not person:
            return None, None
        
        # Check for recent face detection
        if 'face_detections' in person and person['face_detections']:
            latest_face = person['face_detections'][-1]
            return latest_face.get('user_id'), latest_face.get('face_id')
        
        # Check name assignment history
        if person_id in self.name_assignment_history:
            assignment = self.name_assignment_history[person_id]
            return assignment['name'], assignment['face_id']
        
        return None, None

    def get_person_display_name(self, person_id):
        """Get display name for a person"""
        user_id, _ = self.get_person_identity(person_id)
        return user_id if user_id else person_id

    def get_person_user_id(self, person_id):
        """Get user ID for a person"""
        user_id, _ = self.get_person_identity(person_id)
        return user_id

    def get_person_face_id(self, person_id):
        """Get face ID for a person"""
        _, face_id = self.get_person_identity(person_id)
        return face_id