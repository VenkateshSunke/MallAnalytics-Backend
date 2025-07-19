#!/usr/bin/env python3
"""
Test script to verify output path generation for different video formats
"""

import os

def test_output_path_generation():
    """Test the output path generation logic"""
    
    test_cases = [
        "video.mp4",
        "video.mkv", 
        "video.avi",
        "video.mov",
        "video.webm",
        "/path/to/video.mp4",
        "/path/to/video.mkv",
        "video_with_underscores.mp4",
        "video.with.dots.mkv"
    ]
    
    print("Testing output path generation:")
    print("=" * 50)
    
    for test_path in test_cases:
        # Simulate the new logic
        base_name = os.path.splitext(test_path)[0]
        extension = os.path.splitext(test_path)[1]
        processed_path = f"{base_name}_processed{extension}"
        
        print(f"Input:  {test_path}")
        print(f"Output: {processed_path}")
        print("-" * 30)

if __name__ == "__main__":
    test_output_path_generation() 