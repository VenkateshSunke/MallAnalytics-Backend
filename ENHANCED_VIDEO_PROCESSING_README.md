# Enhanced Video Processing System

This document describes the improvements made to the video processing system in `wise_backend/logs/services/videoProcessing.py`.

## 🚀 Key Improvements

### 1. **Fixed Critical Issues**
- ✅ **Removed duplicate function definition** - The `start_process` function was defined twice, causing the second definition to override the first
- ✅ **Removed unused import** - Removed `get_blueprint_info_by_camera_id` which was imported but never used
- ✅ **Enhanced error handling** - Added specific error codes and better error messages for FFmpeg processes

### 2. **Memory Management**
- ✅ **Real-time memory monitoring** - Tracks memory usage throughout processing
- ✅ **Configurable memory limits** - Set memory limits to prevent system crashes
- ✅ **Automatic garbage collection** - Forces cleanup when memory usage is high
- ✅ **Batch processing** - Processes frames in configurable batches to manage memory
- ✅ **Memory usage reporting** - Detailed memory statistics in processing results

### 3. **Performance Optimizations**
- ✅ **Configurable batch sizes** - Adjust batch processing for different performance needs
- ✅ **Enhanced progress logging** - Real-time progress with memory usage information
- ✅ **Performance timing breakdown** - Detailed timing for each processing stage
- ✅ **Memory-aware processing** - Adapts processing based on available memory

### 4. **Enhanced Error Handling**
- ✅ **Specific FFmpeg error codes** - Handles SIGKILL, SIGSEGV, SIGABRT, etc.
- ✅ **Graceful degradation** - Continues processing even if some components fail
- ✅ **Parameter validation** - Validates input parameters with sensible defaults
- ✅ **Resource cleanup** - Ensures proper cleanup of FFmpeg processes

### 5. **Multi-Track Support**
- ✅ **Robust track selection** - Better handling of multiple video tracks
- ✅ **Track validation** - Validates track indices before processing
- ✅ **Track information logging** - Detailed logging of available tracks

## 📋 New Function Parameters

The `start_process` function now accepts additional parameters:

```python
def start_process(camera, output_path, track_index=None, batch_size=10, memory_limit_mb=1024):
```

### Parameters:
- `camera`: Camera configuration object (required)
- `output_path`: Path to video file (required)
- `track_index`: Video track to process (optional, default: first track)
- `batch_size`: Number of frames to process before cleanup (default: 10)
- `memory_limit_mb`: Memory limit in MB (default: 1024)

## 🔧 New Functions Added

### Memory Management Functions:
```python
def get_memory_usage():
    """Get current memory usage in MB"""

def cleanup_memory():
    """Force garbage collection to free memory"""

def check_memory_limit(current_usage_mb, limit_mb):
    """Check if memory usage exceeds limit"""

def validate_processing_parameters(batch_size, memory_limit_mb):
    """Validate processing parameters with sensible defaults"""
```

## 📊 Enhanced Results

The processing results now include detailed memory and performance information:

```python
results = {
    # ... existing fields ...
    'memory_usage': {
        'initial_mb': 150.2,
        'final_mb': 180.5,
        'peak_increase_mb': 30.3,
        'batch_size': 10,
        'memory_limit_mb': 1024
    },
    'performance_timings': {
        'aws_detection': 2.5,
        'yolo_detection': 15.3,
        'tracking_update': 8.7,
        'movement_logging': 1.2,
        'frame_processing': 45.8,
        'memory_management': 0.5
    }
}
```

## 🧪 Testing

### Quick Demo
Run the demonstration script to see the improvements in action:

```bash
cd MallAnalytics-Backend
python demo_enhanced_video_processing.py
```

### Comprehensive Testing
Run the full test suite:

```bash
cd MallAnalytics-Backend
python test_enhanced_video_processing.py
```

## 📈 Performance Examples

### Example 1: Memory-Optimized Processing
```python
result = start_process(
    camera=camera_config,
    output_path="test2.mkv",
    track_index=0,
    batch_size=5,        # Small batches
    memory_limit_mb=512  # Low memory limit
)
```

### Example 2: Performance-Optimized Processing
```python
result = start_process(
    camera=camera_config,
    output_path="test2.mkv",
    track_index=1,
    batch_size=20,       # Large batches
    memory_limit_mb=2048 # High memory limit
)
```

### Example 3: Default Processing
```python
result = start_process(
    camera=camera_config,
    output_path="test2.mkv",
    track_index=2
    # Uses default batch_size=10, memory_limit_mb=1024
)
```

## 🔍 Error Handling Examples

The system now provides detailed error information:

```python
# FFmpeg process errors
if return_code == -9:  # SIGKILL
    logger.error("FFmpeg process was killed due to memory issues")
elif return_code == -11:  # SIGSEGV
    logger.error("FFmpeg process crashed due to segmentation fault")
elif return_code == -6:  # SIGABRT
    logger.error("FFmpeg process aborted")

# Parameter validation
if batch_size < 1:
    logger.warning("Invalid batch_size, using default 10")
if memory_limit_mb < 100:
    logger.warning("Memory limit too low, using minimum 100 MB")
```

## 📦 Dependencies

### New Dependencies Added:
- `psutil==6.1.0` - For memory monitoring

### Updated Requirements:
The `requirements.txt` file has been updated to include the new dependency.

## 🚨 Breaking Changes

### None
The enhanced `start_process` function maintains backward compatibility. All existing code will continue to work without modification.

## 🔧 Configuration

### Memory Management Settings:
- **Minimum memory limit**: 100 MB
- **Maximum memory limit**: 8192 MB (8 GB)
- **Default memory limit**: 1024 MB (1 GB)
- **Default batch size**: 10 frames
- **Memory check interval**: Every 50 frames

### Performance Settings:
- **AWS frame skip**: Every 60 frames (configurable)
- **Progress logging**: Every 100 frames
- **Memory monitoring**: Every 50 frames

## 📝 Logging

Enhanced logging provides detailed information:

```
2024-01-15 10:30:15 - INFO - Processing parameters: batch_size=10, memory_limit=1024 MB
2024-01-15 10:30:15 - INFO - Initial memory usage: 150.2 MB
2024-01-15 10:30:20 - INFO - Progress: 25.0% (181/727) - Speed: 12.1 fps - ETA: 00:45 - Memory: 165.3 MB
2024-01-15 10:30:25 - INFO - Memory limit exceeded (165.3 MB > 1024 MB). Forcing cleanup.
2024-01-15 10:30:30 - INFO - Video processing completed successfully
```

## 🎯 Best Practices

### For Memory-Constrained Systems:
```python
# Use small batches and low memory limits
result = start_process(
    camera=camera_config,
    output_path=video_path,
    batch_size=5,
    memory_limit_mb=512
)
```

### For High-Performance Systems:
```python
# Use large batches and high memory limits
result = start_process(
    camera=camera_config,
    output_path=video_path,
    batch_size=20,
    memory_limit_mb=2048
)
```

### For Production Systems:
```python
# Monitor memory usage and adjust accordingly
result = start_process(
    camera=camera_config,
    output_path=video_path,
    batch_size=10,
    memory_limit_mb=1024
)

# Check memory usage in results
memory_info = result.get('memory_usage', {})
if memory_info.get('peak_increase_mb', 0) > 800:
    logger.warning("High memory usage detected")
```

## 🔮 Future Enhancements

Potential future improvements:
- Parallel processing for multiple tracks
- GPU acceleration support
- Adaptive batch sizing based on system performance
- Real-time memory usage alerts
- Processing pipeline optimization

## 📞 Support

For issues or questions about the enhanced video processing system:
1. Check the logs for detailed error information
2. Review the memory usage statistics
3. Adjust batch size and memory limits as needed
4. Ensure all dependencies are properly installed

---

**Note**: This enhanced system maintains full backward compatibility while providing significant improvements in memory management, error handling, and performance monitoring. 