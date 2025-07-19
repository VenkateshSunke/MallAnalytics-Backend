#!/usr/bin/env python3
"""
Enhanced Video Processing Demo

This script demonstrates the improved videoProcessing.py functionality
using the test2.mkv file with 5 video tracks.
"""

import os
import sys
import time
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def demo_enhanced_video_processing():
    """Demonstrate the enhanced video processing capabilities"""
    
    # Video file path
    video_path = "test2.mkv"
    
    # Check if video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        logger.info("Please ensure test2.mkv is in the current directory")
        return
    
    # Camera configuration
    camera_config = {
        'id': 'demo_camera_001',
        'name': 'Demo Camera',
        'stores': {
            'store1': {
                'name': 'Demo Store 1',
                'polygon': [[100, 100], [300, 100], [300, 300], [100, 300]]
            },
            'store2': {
                'name': 'Demo Store 2',
                'polygon': [[400, 100], [600, 100], [600, 300], [400, 300]]
            }
        }
    }
    
    logger.info("Enhanced Video Processing Demo")
    logger.info("="*50)
    logger.info(f"Video file: {video_path}")
    logger.info(f"Camera ID: {camera_config['id']}")
    logger.info(f"Number of stores: {len(camera_config['stores'])}")
    
    try:
        from wise_backend.logs.services.videoProcessing import start_process
        
        # Demo 1: Process with default settings
        logger.info("\nDemo 1: Processing with default settings")
        logger.info("-" * 30)
        
        start_time = time.time()
        result1 = start_process(
            camera=camera_config,
            output_path=video_path,
            track_index=0  # First track
        )
        processing_time1 = time.time() - start_time
        
        logger.info(f"✓ Default processing completed in {processing_time1:.2f} seconds")
        logger.info(f"  - Frames processed: {result1.get('processed_frames', 0)}")
        logger.info(f"  - Processing speed: {result1.get('processing_speed', 0):.2f} fps")
        logger.info(f"  - Memory usage: {result1.get('memory_usage', {}).get('peak_increase_mb', 0):.1f} MB")
        
        # Demo 2: Process with memory optimization
        logger.info("\nDemo 2: Processing with memory optimization")
        logger.info("-" * 30)
        
        start_time = time.time()
        result2 = start_process(
            camera=camera_config,
            output_path=video_path,
            track_index=1,  # Second track
            batch_size=5,   # Smaller batch size
            memory_limit_mb=512  # Lower memory limit
        )
        processing_time2 = time.time() - start_time
        
        logger.info(f"✓ Memory-optimized processing completed in {processing_time2:.2f} seconds")
        logger.info(f"  - Frames processed: {result2.get('processed_frames', 0)}")
        logger.info(f"  - Processing speed: {result2.get('processing_speed', 0):.2f} fps")
        logger.info(f"  - Memory usage: {result2.get('memory_usage', {}).get('peak_increase_mb', 0):.1f} MB")
        
        # Demo 3: Process with performance optimization
        logger.info("\nDemo 3: Processing with performance optimization")
        logger.info("-" * 30)
        
        start_time = time.time()
        result3 = start_process(
            camera=camera_config,
            output_path=video_path,
            track_index=2,  # Third track
            batch_size=20,  # Larger batch size
            memory_limit_mb=2048  # Higher memory limit
        )
        processing_time3 = time.time() - start_time
        
        logger.info(f"✓ Performance-optimized processing completed in {processing_time3:.2f} seconds")
        logger.info(f"  - Frames processed: {result3.get('processed_frames', 0)}")
        logger.info(f"  - Processing speed: {result3.get('processing_speed', 0):.2f} fps")
        logger.info(f"  - Memory usage: {result3.get('memory_usage', {}).get('peak_increase_mb', 0):.1f} MB")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("DEMO SUMMARY")
        logger.info("="*50)
        logger.info(f"Default processing:     {processing_time1:.2f}s, {result1.get('processing_speed', 0):.2f} fps")
        logger.info(f"Memory optimized:       {processing_time2:.2f}s, {result2.get('processing_speed', 0):.2f} fps")
        logger.info(f"Performance optimized:  {processing_time3:.2f}s, {result3.get('processing_speed', 0):.2f} fps")
        
        # Performance comparison
        speeds = [
            result1.get('processing_speed', 0),
            result2.get('processing_speed', 0),
            result3.get('processing_speed', 0)
        ]
        
        memory_usage = [
            result1.get('memory_usage', {}).get('peak_increase_mb', 0),
            result2.get('memory_usage', {}).get('peak_increase_mb', 0),
            result3.get('memory_usage', {}).get('peak_increase_mb', 0)
        ]
        
        logger.info(f"\nBest processing speed: {max(speeds):.2f} fps")
        logger.info(f"Lowest memory usage: {min(memory_usage):.1f} MB")
        logger.info(f"Average processing speed: {sum(speeds)/len(speeds):.2f} fps")
        
        logger.info("\n✓ Enhanced video processing demo completed successfully!")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure you're running this from the MallAnalytics-Backend directory")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.info("Check the logs for more details")

def demo_error_handling():
    """Demonstrate error handling capabilities"""
    logger.info("\nError Handling Demo")
    logger.info("="*50)
    
    try:
        from wise_backend.logs.services.videoProcessing import start_process
        
        # Test with invalid parameters
        logger.info("Testing invalid camera configuration...")
        try:
            start_process(camera=None, output_path="test.mkv")
        except ValueError as e:
            logger.info(f"✓ Correctly caught error: {e}")
        
        logger.info("Testing invalid output path...")
        try:
            start_process(camera={'id': 'test'}, output_path=None)
        except ValueError as e:
            logger.info(f"✓ Correctly caught error: {e}")
        
        logger.info("Testing non-existent file...")
        try:
            start_process(camera={'id': 'test'}, output_path="non_existent.mkv")
        except FileNotFoundError as e:
            logger.info(f"✓ Correctly caught error: {e}")
        
        logger.info("✓ Error handling demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error handling demo failed: {e}")

if __name__ == "__main__":
    logger.info("Starting Enhanced Video Processing Demo")
    
    # Run the main demo
    demo_enhanced_video_processing()
    
    # Run error handling demo
    demo_error_handling()
    
    logger.info("\nDemo completed!") 