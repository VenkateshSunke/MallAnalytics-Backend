#!/usr/bin/env python3
"""
FFmpeg Diagnosis Script

This script helps diagnose FFmpeg-related issues in the video processing system.
Run this script to check FFmpeg availability and test basic functionality.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_ffmpeg_installation():
    """Check if FFmpeg is installed and accessible"""
    logger.info("=== FFmpeg Installation Check ===")
    
    try:
        # Check if ffmpeg command exists
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown version"
            logger.info(f"✅ FFmpeg is installed: {version_line}")
            return True
        else:
            logger.error(f"❌ FFmpeg command failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.error("❌ FFmpeg not found in system PATH")
        logger.info("💡 To install FFmpeg:")
        logger.info("   - Windows: Download from https://ffmpeg.org/download.html")
        logger.info("   - macOS: brew install ffmpeg")
        logger.info("   - Ubuntu/Debian: sudo apt install ffmpeg")
        logger.info("   - CentOS/RHEL: sudo yum install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        logger.error("❌ FFmpeg command timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking FFmpeg: {e}")
        return False

def check_python_ffmpeg_package():
    """Check if the python-ffmpeg package is installed"""
    logger.info("\n=== Python FFmpeg Package Check ===")
    
    try:
        import ffmpeg
        # Try to get version, but don't fail if it doesn't exist
        try:
            version = ffmpeg.__version__
            logger.info(f"✅ python-ffmpeg package is installed: {version}")
        except AttributeError:
            logger.info("✅ python-ffmpeg package is installed (version info not available)")
        return True
    except ImportError:
        logger.error("❌ python-ffmpeg package not installed")
        logger.info("💡 Install with: pip install ffmpeg-python")
        return False
    except Exception as e:
        logger.error(f"❌ Error importing ffmpeg: {e}")
        return False

def test_ffmpeg_basic_functionality():
    """Test basic FFmpeg functionality"""
    logger.info("\n=== FFmpeg Basic Functionality Test ===")
    
    try:
        import ffmpeg
        
        # Test creating a simple ffmpeg command
        input_stream = ffmpeg.input('pipe:')
        output_stream = input_stream.output('pipe:', format='rawvideo', pix_fmt='bgr24')
        
        logger.info("✅ Successfully created FFmpeg command structure")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing FFmpeg functionality: {e}")
        return False

def check_system_environment():
    """Check system environment variables"""
    logger.info("\n=== System Environment Check ===")
    
    # Check PATH
    path = os.environ.get('PATH', '')
    logger.info(f"PATH environment variable length: {len(path)} characters")
    
    # Check if ffmpeg is in PATH
    path_dirs = path.split(os.pathsep)
    ffmpeg_found = False
    
    for directory in path_dirs:
        if os.path.exists(os.path.join(directory, 'ffmpeg')):
            logger.info(f"✅ Found ffmpeg in: {directory}")
            ffmpeg_found = True
            break
        elif os.path.exists(os.path.join(directory, 'ffmpeg.exe')):
            logger.info(f"✅ Found ffmpeg.exe in: {directory}")
            ffmpeg_found = True
            break
    
    if not ffmpeg_found:
        logger.warning("⚠️  FFmpeg not found in PATH directories")
    
    # Check Python environment
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    
    return ffmpeg_found

def test_video_file_processing(video_path=None):
    """Test processing a video file"""
    logger.info("\n=== Video File Processing Test ===")
    
    if not video_path:
        logger.warning("⚠️  No video file provided for testing")
        logger.info("💡 To test with a video file, run: python test_ffmpeg_diagnosis.py <video_file_path>")
        return False
    
    if not os.path.exists(video_path):
        logger.error(f"❌ Video file not found: {video_path}")
        return False
    
    logger.info(f"Testing with video file: {video_path}")
    
    try:
        import ffmpeg
        
        # Get video info
        probe = ffmpeg.probe(video_path)
        logger.info(f"✅ Successfully probed video file")
        
        # Get video streams
        video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
        logger.info(f"Found {len(video_streams)} video stream(s)")
        
        for i, stream in enumerate(video_streams):
            logger.info(f"  Stream {i}: {stream.get('width')}x{stream.get('height')} "
                       f"({stream.get('codec_name', 'unknown')})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing video file: {e}")
        return False

def main():
    """Main diagnostic function"""
    logger.info("🔍 FFmpeg Diagnosis Tool")
    logger.info("=" * 50)
    
    # Get video file path from command line argument
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run all checks
    checks = [
        ("FFmpeg Installation", check_ffmpeg_installation),
        ("Python FFmpeg Package", check_python_ffmpeg_package),
        ("FFmpeg Functionality", test_ffmpeg_basic_functionality),
        ("System Environment", check_system_environment),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"❌ Error in {check_name}: {e}")
            results[check_name] = False
    
    # Test video processing if file provided
    if video_path:
        results["Video Processing"] = test_video_file_processing(video_path)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 DIAGNOSIS SUMMARY")
    logger.info("=" * 50)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("\n" + "=" * 50)
    if all_passed:
        logger.info("🎉 All checks passed! FFmpeg should work correctly.")
    else:
        logger.info("⚠️  Some checks failed. Please address the issues above.")
        logger.info("\n💡 Common solutions:")
        logger.info("1. Install FFmpeg: https://ffmpeg.org/download.html")
        logger.info("2. Add FFmpeg to your system PATH")
        logger.info("3. Install python-ffmpeg: pip install ffmpeg-python")
        logger.info("4. Restart your terminal/IDE after installation")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 