# FFmpeg Video Processing

This document describes the updated video processing functionality that uses FFmpeg instead of OpenCV for reading video files, particularly useful for MKV files with multiple tracks.

## Overview

The video processing system has been updated to use `ffmpeg-python` for reading video files, which provides better support for:
- MKV files with multiple video tracks
- Various video codecs
- Better error handling and stream selection

## New Features

### 1. Track Selection
You can now specify which video track to process when dealing with multi-track video files.

### 2. Video Information API
A new endpoint to get detailed information about video files including available tracks.

### 3. Enhanced Error Handling
Better error handling for video file reading and track selection.

## API Endpoints

### 1. Get Video Information
**POST** `/api/logs/video-info/`

Get detailed information about a video file including available tracks.

**Request Body:**
```json
{
    "video_path": "/path/to/your/video.mkv"
}
```

**Response:**
```json
{
    "message": "Video information retrieved successfully",
    "video_info": {
        "format": {...},
        "streams": [...],
        "video_tracks": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "duration": "120.5",
                "bit_rate": "5000000",
                "language": "eng"
            }
        ],
        "audio_tracks": [...],
        "subtitle_tracks": [...]
    },
    "video_path": "/path/to/your/video.mkv"
}
```

### 2. Process Video with Track Selection
**POST** `/api/logs/test-video-processing/`

Process a video file with optional track selection.

**Request Body:**
```json
{
    "video_path": "/path/to/your/video.mkv",
    "video_track_index": 0,
    "camera_id": "camera_001",
    "camera_name": "Test Camera"
}
```

**Parameters:**
- `video_path` (required): Path to the video file
- `video_track_index` (optional): Index of the video track to process (default: 0)
- `camera_id` (optional): Camera identifier
- `camera_name` (optional): Camera name

**Response:**
```json
{
    "message": "Video processing completed successfully",
    "results": {
        "total_frames": 3600,
        "processed_frames": 3600,
        "processing_time": 45.2,
        "video_track_used": 0,
        "video_info": {...},
        "ffmpeg_used": true,
        ...
    },
    "camera_config": {...}
}
```

## Usage Examples

### Python Script Example

```python
import requests

# Get video information first
video_info_response = requests.post('http://localhost:8000/api/logs/video-info/', {
    'video_path': '/path/to/video.mkv'
})

if video_info_response.status_code == 200:
    video_info = video_info_response.json()['video_info']
    print(f"Available video tracks: {len(video_info['video_tracks'])}")
    
    # Process with specific track
    for i, track in enumerate(video_info['video_tracks']):
        print(f"Processing track {i}: {track['width']}x{track['height']}")
        
        processing_response = requests.post('http://localhost:8000/api/logs/test-video-processing/', {
            'video_path': '/path/to/video.mkv',
            'video_track_index': i,
            'camera_id': f'camera_{i}',
            'camera_name': f'Camera {i}'
        })
        
        if processing_response.status_code == 200:
            results = processing_response.json()['results']
            print(f"Track {i} processed: {results['processed_frames']} frames")
```

### Command Line Example

```bash
# Get video information
curl -X POST http://localhost:8000/api/logs/video-info/ \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mkv"}'

# Process video with track 0
curl -X POST http://localhost:8000/api/logs/test-video-processing/ \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mkv",
    "video_track_index": 0,
    "camera_id": "camera_001"
  }'
```

## Installation

Make sure you have the required dependencies:

```bash
pip install ffmpeg-python
```

The `ffmpeg-python` package is already included in the `requirements.txt` file.

## Testing

Use the provided test script to verify functionality:

```bash
cd MallAnalytics-Backend
python test_ffmpeg_video.py
```

## Technical Details

### Key Changes

1. **Video Reading**: Replaced OpenCV's `cv2.VideoCapture` with FFmpeg's async process
2. **Track Selection**: Added support for selecting specific video tracks
3. **Frame Processing**: Raw frame data is read from FFmpeg and converted to numpy arrays
4. **Error Handling**: Enhanced error handling for video file operations

### Performance Considerations

- FFmpeg provides better performance for large video files
- Async processing reduces memory usage
- Track selection allows processing only the needed video stream

### Limitations

- Requires FFmpeg to be installed on the system
- Currently supports BGR24 pixel format (compatible with OpenCV)
- Frame rate detection may be approximate for some video formats

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is installed and accessible in PATH
2. **Track index out of range**: Check available tracks using the video-info endpoint
3. **Memory issues**: Large video files may require sufficient RAM

### Error Messages

- `"Could not get video information"`: Check if video file exists and is readable
- `"Could not start ffmpeg process"`: Verify FFmpeg installation
- `"FFmpeg process terminated unexpectedly"`: Check video file integrity

## Migration from OpenCV

The API remains backward compatible. Existing code will continue to work, but you can now specify track indices for multi-track video files. 