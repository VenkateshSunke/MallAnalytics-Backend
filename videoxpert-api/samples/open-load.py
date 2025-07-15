import cv2
import os

def play_video_robust(video_path):
    """
    Play video with robust error handling for corrupted files
    """
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist")
        return False
    
    # Try different backends
    backends = [
        cv2.CAP_ANY,
        cv2.CAP_DSHOW,  # Windows
        cv2.CAP_V4L2,   # Linux
        cv2.CAP_GSTREAMER,
        cv2.CAP_FFMPEG,
        
    ]
    
    cap = None
    for backend, index in enumerate(backends):
        try:
            cap = cv2.VideoCapture(video_path, backend)
            if cap.isOpened():
                print(f"Successfully opened with backend: {backend}, index: {index}")
                break
            else:
                cap.release()
        except Exception as e:
            print(f"Failed with backend {backend}: {e}")
            continue
        
 
    
    if cap is None or not cap.isOpened():
        print(f"Error: Cannot open video file {video_path} with any backend")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"Video properties:")
    print(f"  Resolution: {int(width)}x{int(height)}")
    print(f"  FPS: {fps}")
    print(f"  Frame count: {int(frame_count)}")
    print(f"  Duration: {frame_count/fps:.2f} seconds")
    
    frame_num = 0
    successful_frames = 0
    failed_frames = 0
    
    print(f"Playing video: {video_path}")
    print("Press 'q' to quit, 's' to skip corrupted frames")
    
    while True:
        ret, frame = cap.read()
        frame_num += 1
        
        if not ret:
            if frame_num < frame_count:
                print(f"Failed to read frame {frame_num}")
                failed_frames += 1
                # Try to skip to next frame
                continue
            else:
                print("Reached end of video.")
                break
        
        successful_frames += 1
        
        cv2.imshow('Video Playback', frame)
        
        # Adjust wait time based on FPS
        wait_time = max(1, int(1000/fps)) if fps > 0 else 25
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            print("Playback interrupted by user.")
            break
        elif key == ord('s'):
            print(f"Skipping... Current frame: {frame_num}")
            continue
    
    print(f"Playback finished. Successfully read {successful_frames} frames, failed on {failed_frames} frames")
    cap.release()
    cv2.destroyAllWindows()
    return True

# Path to your video file
video_path = "./1# CCTV050-P1 Pasadizo Joaquin M.@2025-07-11T102956.567Z.mkv"

# Play the video
play_video_robust(video_path)