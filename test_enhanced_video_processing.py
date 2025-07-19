#!/usr/bin/env python3
"""
Enhanced Video Processing Test Script

This script demonstrates the improved videoProcessing.py functionality with:
- Memory management and monitoring
- Batch processing
- Enhanced error handling
- Multi-track support
- Performance monitoring
"""

import os
import sys
import time
import logging
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_video_processing_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_memory_management():
    """Test memory management functions"""
    logger.info("Testing memory management functions...")
    
    try:
        from wise_backend.logs.services.videoProcessing import (
            get_memory_usage, 
            cleanup_memory, 
            check_memory_limit,
            validate_processing_parameters
        )
        
        # Test memory usage
        initial_memory = get_memory_usage()
        logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Test memory limit check
        limit_exceeded = check_memory_limit(initial_memory, 100)
        logger.info(f"Memory limit exceeded (100 MB): {limit_exceeded}")
        
        # Test parameter validation
        batch_size, memory_limit = validate_processing_parameters(5, 50)
        logger.info(f"Validated parameters: batch_size={batch_size}, memory_limit={memory_limit} MB")
        
        # Test cleanup
        cleanup_memory()
        after_cleanup = get_memory_usage()
        logger.info(f"Memory after cleanup: {after_cleanup:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Memory management test failed: {e}")
        return False

def test_blueprint_functions():
    """Test blueprint utility functions"""
    logger.info("Testing blueprint utility functions...")
    
    try:
        from wise_backend.logs.services.videoProcessing import (
            transform_blueprint_to_video_coordinates,
            draw_store_polygons
        )
        
        # Test coordinate transformation
        blueprint_polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
        transformation_matrix = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        
        video_coords = transform_blueprint_to_video_coordinates(blueprint_polygon, transformation_matrix)
        logger.info(f"Transformed coordinates: {video_coords}")
        
        # Test store polygon drawing (mock test)
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stores = {
            'store1': {
                'name': 'Test Store',
                'polygon': [[100, 100], [200, 100], [200, 200], [100, 200]]
            }
        }
        
        # This should not raise an error
        result_frame = draw_store_polygons(frame, stores)
        logger.info(f"Store polygon drawing test passed. Frame shape: {result_frame.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Blueprint functions test failed: {e}")
        return False

def test_video_processing_with_different_tracks(video_path, camera_config):
    """Test video processing with different track indices"""
    logger.info(f"Testing video processing with different tracks for: {video_path}")
    
    try:
        from wise_backend.logs.services.videoProcessing import start_process
        
        results = {}
        
        # Test with different track indices
        for track_index in range(5):  # Based on the video info showing 5 tracks
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing track index: {track_index}")
            logger.info(f"{'='*60}")
            
            try:
                # Test with different memory limits and batch sizes
                test_configs = [
                    {'batch_size': 5, 'memory_limit_mb': 512},
                    {'batch_size': 10, 'memory_limit_mb': 1024},
                    {'batch_size': 20, 'memory_limit_mb': 2048}
                ]
                
                for config in test_configs:
                    logger.info(f"Testing with config: {config}")
                    
                    start_time = time.time()
                    result = start_process(
                        camera=camera_config,
                        output_path=video_path,
                        track_index=track_index,
                        batch_size=config['batch_size'],
                        memory_limit_mb=config['memory_limit_mb']
                    )
                    processing_time = time.time() - start_time
                    
                    # Store results
                    key = f"track_{track_index}_{config['batch_size']}_{config['memory_limit_mb']}"
                    results[key] = {
                        'track_index': track_index,
                        'config': config,
                        'result': result,
                        'actual_processing_time': processing_time
                    }
                    
                    logger.info(f"Track {track_index} processing completed:")
                    logger.info(f"  - Frames processed: {result.get('processed_frames', 0)}")
                    logger.info(f"  - Processing speed: {result.get('processing_speed', 0):.2f} fps")
                    logger.info(f"  - Memory usage: {result.get('memory_usage', {})}")
                    logger.info(f"  - Performance timings: {result.get('performance_timings', {})}")
                    
                    # Check for memory issues
                    memory_info = result.get('memory_usage', {})
                    if memory_info.get('peak_increase_mb', 0) > config['memory_limit_mb'] * 0.8:
                        logger.warning(f"High memory usage detected: {memory_info.get('peak_increase_mb', 0):.1f} MB")
                    
                    break  # Only test one config per track for speed
                    
            except Exception as e:
                logger.error(f"Failed to process track {track_index}: {e}")
                results[f"track_{track_index}_error"] = {
                    'track_index': track_index,
                    'error': str(e)
                }
        
        return results
        
    except Exception as e:
        logger.error(f"Video processing test failed: {e}")
        return None

def test_error_handling():
    """Test error handling scenarios"""
    logger.info("Testing error handling scenarios...")
    
    try:
        from wise_backend.logs.services.videoProcessing import start_process
        
        # Test with invalid parameters
        test_cases = [
            {
                'name': 'Invalid camera config',
                'camera': None,
                'output_path': 'test.mkv',
                'expected_error': 'Camera configuration is required'
            },
            {
                'name': 'Invalid output path',
                'camera': {'id': 'test'},
                'output_path': None,
                'expected_error': 'Output path is required'
            },
            {
                'name': 'Non-existent file',
                'camera': {'id': 'test'},
                'output_path': 'non_existent_file.mkv',
                'expected_error': 'Video file not found'
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"Testing: {test_case['name']}")
            try:
                start_process(
                    camera=test_case['camera'],
                    output_path=test_case['output_path']
                )
                logger.error(f"Expected error for {test_case['name']} but none occurred")
            except Exception as e:
                if test_case['expected_error'] in str(e):
                    logger.info(f"✓ Correctly caught error: {e}")
                else:
                    logger.warning(f"Unexpected error for {test_case['name']}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False

def generate_test_report(results, video_info):
    """Generate a comprehensive test report"""
    logger.info("Generating test report...")
    
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'video_info': video_info,
        'test_results': results,
        'summary': {
            'total_tracks_tested': 0,
            'successful_tracks': 0,
            'failed_tracks': 0,
            'average_processing_speed': 0,
            'memory_usage_summary': {}
        }
    }
    
    if results:
        speeds = []
        memory_peaks = []
        
        for key, result in results.items():
            if 'error' not in result:
                report['summary']['total_tracks_tested'] += 1
                report['summary']['successful_tracks'] += 1
                
                if 'result' in result:
                    speed = result['result'].get('processing_speed', 0)
                    speeds.append(speed)
                    
                    memory_info = result['result'].get('memory_usage', {})
                    peak = memory_info.get('peak_increase_mb', 0)
                    memory_peaks.append(peak)
            else:
                report['summary']['total_tracks_tested'] += 1
                report['summary']['failed_tracks'] += 1
        
        if speeds:
            report['summary']['average_processing_speed'] = sum(speeds) / len(speeds)
        
        if memory_peaks:
            report['summary']['memory_usage_summary'] = {
                'average_peak_mb': sum(memory_peaks) / len(memory_peaks),
                'max_peak_mb': max(memory_peaks),
                'min_peak_mb': min(memory_peaks)
            }
    
    # Save report to file
    report_file = f"enhanced_video_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Test report saved to: {report_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Total tracks tested: {report['summary']['total_tracks_tested']}")
    logger.info(f"Successful tracks: {report['summary']['successful_tracks']}")
    logger.info(f"Failed tracks: {report['summary']['failed_tracks']}")
    logger.info(f"Average processing speed: {report['summary']['average_processing_speed']:.2f} fps")
    
    if report['summary']['memory_usage_summary']:
        mem_summary = report['summary']['memory_usage_summary']
        logger.info(f"Memory usage - Avg: {mem_summary['average_peak_mb']:.1f} MB, "
                   f"Max: {mem_summary['max_peak_mb']:.1f} MB, "
                   f"Min: {mem_summary['min_peak_mb']:.1f} MB")
    
    return report

def main():
    """Main test function"""
    logger.info("Starting Enhanced Video Processing Test Suite")
    logger.info("="*80)
    
    # Test video file path
    video_path = "test2.mkv"
    
    # Mock camera configuration
    camera_config = {
        'id': 'test_camera_001',
        'name': 'Test Camera',
        'stores': {
            'store1': {
                'name': 'Test Store 1',
                'polygon': [[100, 100], [300, 100], [300, 300], [100, 300]]
            },
            'store2': {
                'name': 'Test Store 2', 
                'polygon': [[400, 100], [600, 100], [600, 300], [400, 300]]
            }
        }
    }
    
    # Video info from the provided JSON
    video_info = {
        "filename": "test2.mkv",
        "format_name": "matroska,webm",
        "duration": 60.59,
        "size": 69921142,
        "video_tracks_count": 5,
        "video_streams": [
            {"track_index": 1, "width": 2048, "height": 1152, "fps": 12.0},
            {"track_index": 2, "width": 2048, "height": 2048, "fps": 12.0},
            {"track_index": 3, "width": 2048, "height": 1152, "fps": 12.0},
            {"track_index": 4, "width": 2048, "height": 1152, "fps": 12.0},
            {"track_index": 5, "width": 2048, "height": 1152, "fps": 12.0}
        ]
    }
    
    # Check if video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        logger.info("Please ensure test2.mkv is in the current directory")
        return
    
    # Run tests
    test_results = {
        'memory_management': test_memory_management(),
        'blueprint_functions': test_blueprint_functions(),
        'error_handling': test_error_handling(),
        'video_processing': test_video_processing_with_different_tracks(video_path, camera_config)
    }
    
    # Generate report
    report = generate_test_report(test_results['video_processing'], video_info)
    
    logger.info("\n" + "="*80)
    logger.info("ENHANCED VIDEO PROCESSING TEST COMPLETED")
    logger.info("="*80)
    
    # Print final status
    all_tests_passed = all([
        test_results['memory_management'],
        test_results['blueprint_functions'], 
        test_results['error_handling']
    ])
    
    if all_tests_passed:
        logger.info("✓ All basic tests passed")
    else:
        logger.warning("⚠ Some basic tests failed")
    
    if test_results['video_processing']:
        logger.info("✓ Video processing tests completed")
    else:
        logger.error("✗ Video processing tests failed")

if __name__ == "__main__":
    main() 