#!/usr/bin/env python3
"""
Test script for ffmpeg video processing functionality
"""

import os
import sys
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wise_backend.settings')
django.setup()

from wise_backend.logs.services.videoProcessing import get_video_info, start_process

def test_video_info(video_path):
    """Test getting video information"""
    print(f"Testing video info for: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    try:
        video_info = get_video_info(video_path)
        if video_info:
            print("✓ Video info retrieved successfully")
            print(f"  Video tracks: {len(video_info['video_tracks'])}")
            print(f"  Audio tracks: {len(video_info['audio_tracks'])}")
            print(f"  Subtitle tracks: {len(video_info['subtitle_tracks'])}")
            
            for i, track in enumerate(video_info['video_tracks']):
                print(f"    Track {i}: {track['width']}x{track['height']} ({track['codec_name']})")
            
            return True
        else:
            print("✗ Failed to get video info")
            return False
    except Exception as e:
        print(f"✗ Error getting video info: {e}")
        return False

def test_video_processing(video_path, track_index=0):
    """Test video processing with a specific track"""
    print(f"\nTesting video processing for track {track_index}")
    
    # Create a simple camera config
    camera_config = {
        'id': 'test_camera_001',
        'name': 'Test Camera',
        'stores': {
            'store_001': {
                'name': 'Test Store 1',
                'video_polygon': [[100, 100], [300, 100], [300, 300], [100, 300]],
                'is_mapped': True
            }
        }
    }
    
    try:
        results = start_process(camera_config, video_path, track_index)
        print("✓ Video processing completed successfully")
        print(f"  Processed frames: {results['processed_frames']}")
        print(f"  Processing time: {results['processing_time']:.2f}s")
        print(f"  Track used: {results['video_track_used']}")
        return True
    except Exception as e:
        print(f"✗ Error in video processing: {e}")
        return False

if __name__ == "__main__":
    # Test with a sample video file
    video_path = "yoohoo.mkv"  # Adjust this path as needed
    
    print("FFmpeg Video Processing Test")
    print("=" * 40)
    
    # Test video info
    if test_video_info(video_path):
        # Test processing with first track
        test_video_processing(video_path, 0)
        
        # If multiple tracks exist, test with second track
        video_info = get_video_info(video_path)
        if video_info and len(video_info['video_tracks']) > 1:
            test_video_processing(video_path, 1)
    else:
        print("Skipping video processing test due to video info failure") 