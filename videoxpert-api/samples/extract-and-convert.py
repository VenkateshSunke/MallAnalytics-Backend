"""
Ultra-Fast VideoXpert Export Processor
Parallel processing with hardware acceleration for massive video batches
"""

import os
import zipfile
import subprocess
import re
import sys
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import threading
from pathlib import Path

def get_hardware_info():
    """Detect available hardware acceleration."""
    hw_options = []
    
    # Test for NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            hw_options.append('nvenc')
            print("✓ NVIDIA GPU detected - will use NVENC hardware encoding")
    except:
        pass
    
    # Test for Intel Quick Sync
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=5)
        if 'h264_qsv' in result.stdout:
            hw_options.append('qsv')
            print("✓ Intel Quick Sync detected")
    except:
        pass
    
    # Test for AMD VCE
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=5)
        if 'h264_amf' in result.stdout:
            hw_options.append('amf')
            print("✓ AMD VCE detected")
    except:
        pass
    
    if not hw_options:
        print("⚠ No hardware acceleration detected - using CPU (will be slower)")
        hw_options.append('cpu')
    
    return hw_options

def get_optimal_settings():
    """Get optimal encoding settings for speed."""
    cpu_count = multiprocessing.cpu_count()
    
    settings = {
        'max_workers': min(cpu_count, 8),  # Don't exceed 8 parallel jobs
        'threads_per_job': max(32, cpu_count // 4),  # Threads per ffmpeg process
        'hw_accel': get_hardware_info()[0]  # Use best available
    }
    
    print(f"CPU cores: {cpu_count}")
    print(f"Parallel jobs: {settings['max_workers']}")
    print(f"Threads per job: {settings['threads_per_job']}")
    print(f"Hardware acceleration: {settings['hw_accel']}")
    
    return settings

def extract_mkv_from_zip(zip_path):
    """Extract MKV file from ZIP archive."""
    output_dir = os.path.dirname(zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        mkv_files = [f for f in zip_ref.namelist() if f.lower().endswith('.mkv')]
        
        if not mkv_files:
            return None
        
        mkv_file = mkv_files[0]
        zip_ref.extract(mkv_file, output_dir)
        extracted_path = os.path.join(output_dir, mkv_file)
        
        if os.path.dirname(mkv_file):
            new_path = os.path.join(output_dir, os.path.basename(mkv_file))
            os.rename(extracted_path, new_path)
            extracted_path = new_path
        
        return extracted_path

def get_video_duration(video_path):
    """Get video duration quickly."""
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except:
        return None

def convert_ultra_fast(mkv_file, output_file, hw_accel='nvenc', threads=4):
    """Ultra-fast conversion with hardware acceleration."""
    try:
        # Different command based on hardware
        if hw_accel == 'nvenc':
            cmd = [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',
                '-i', mkv_file,
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',  # Fastest NVENC preset
                '-tune', 'hq',
                '-rc', 'vbr',
                '-cq', '28',  # Lower = better quality, higher = faster
                '-c:a', 'aac', '-b:a', '128k',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts+discardcorrupt',
                '-err_detect', 'ignore_err',
                output_file
            ]
        elif hw_accel == 'qsv':
            cmd = [
                'ffmpeg', '-y',
                '-hwaccel', 'qsv',
                '-i', mkv_file,
                '-c:v', 'h264_qsv',
                '-preset', 'veryfast',
                '-global_quality', '28',
                '-c:a', 'aac', '-b:a', '128k',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts+discardcorrupt',
                '-err_detect', 'ignore_err',
                output_file
            ]
        elif hw_accel == 'amf':
            cmd = [
                'ffmpeg', '-y',
                '-i', mkv_file,
                '-c:v', 'h264_amf',
                '-quality', 'speed',
                '-rc', 'vbr_peak',
                '-qp_i', '28',
                '-c:a', 'aac', '-b:a', '128k',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts+discardcorrupt',
                '-err_detect', 'ignore_err',
                output_file
            ]
        else:  # CPU fallback
            cmd = [
                'ffmpeg', '-y',
                '-i', mkv_file,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-crf', '28',
                '-threads', str(threads),
                '-c:a', 'aac', '-b:a', '128k',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts+discardcorrupt',
                '-err_detect', 'ignore_err',
                output_file
            ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        end_time = time.time()
        
        if result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
            duration = get_video_duration(mkv_file)
            conversion_time = end_time - start_time
            speed = f"{duration/conversion_time:.1f}x" if duration and conversion_time > 0 else "unknown"
            return output_file, conversion_time, speed
        else:
            return None, 0, "failed"
            
    except Exception as e:
        return None, 0, f"error: {e}"

def process_single_file(zip_path, settings):
    """Process a single ZIP file."""
    start_time = time.time()
    filename = os.path.basename(zip_path)
    
    try:
        # Extract MKV
        mkv_path = extract_mkv_from_zip(zip_path)
        if not mkv_path:
            return {
                'file': filename,
                'status': 'failed',
                'error': 'No MKV found in ZIP',
                'time': 0
            }
        
        # Convert to MP4
        input_filename = os.path.basename(mkv_path)
        output_filename = os.path.splitext(input_filename)[0] + ".mp4"
        output_path = os.path.dirname(mkv_path)
        output_file = os.path.join(output_path, output_filename)
        
        result, conv_time, speed = convert_ultra_fast(
            mkv_path, 
            output_file, 
            settings['hw_accel'], 
            settings['threads_per_job']
        )
        
        # Cleanup
        try:
            os.remove(mkv_path)
        except:
            pass
        
        total_time = time.time() - start_time
        
        if result:
            return {
                'file': filename,
                'status': 'success',
                'output': result,
                'time': total_time,
                'speed': speed,
                'conversion_time': conv_time
            }
        else:
            return {
                'file': filename,
                'status': 'failed',
                'error': 'Conversion failed',
                'time': total_time
            }
            
    except Exception as e:
        return {
            'file': filename,
            'status': 'failed',
            'error': str(e),
            'time': time.time() - start_time
        }

def process_batch(zip_files, max_workers=None):
    """Process multiple ZIP files in parallel."""
    if not zip_files:
        print("No ZIP files provided")
        return
    
    settings = get_optimal_settings()
    if max_workers:
        settings['max_workers'] = max_workers
    
    print(f"\n🚀 Starting batch processing of {len(zip_files)} files...")
    print(f"⚡ Using {settings['max_workers']} parallel workers")
    print("=" * 60)
    
    results = []
    completed = 0
    failed = 0
    total_start = time.time()
    
    with ThreadPoolExecutor(max_workers=settings['max_workers']) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_file, zip_file, settings): zip_file 
            for zip_file in zip_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                completed += 1
                print(f"✅ [{completed}/{len(zip_files)}] {result['file']} "
                      f"({result['time']:.1f}s, {result['speed']} speed)")
            else:
                failed += 1
                print(f"❌ [{completed+failed}/{len(zip_files)}] {result['file']} "
                      f"FAILED: {result.get('error', 'Unknown error')}")
    
    # Final summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"🏁 BATCH COMPLETE!")
    print(f"✅ Successful: {completed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Total time: {total_time:.1f} seconds")
    print(f"⚡ Average per file: {total_time/len(zip_files):.1f} seconds")
    
    if completed > 0:
        avg_speed = sum(float(r['speed'].replace('x', '')) for r in results 
                       if r['status'] == 'success' and 'x' in r['speed']) / completed
        print(f"🚀 Average conversion speed: {avg_speed:.1f}x realtime")

def find_zip_files(directory):
    """Find all ZIP files in a directory."""
    zip_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.zip'):
                zip_files.append(os.path.join(root, file))
    return zip_files

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python ultra_fast_converter.py <zip_file>")
        print("  Batch mode:  python ultra_fast_converter.py <directory> [max_workers]")
        sys.exit(1)
    
    path = sys.argv[1]
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if os.path.isfile(path):
        # Single file mode
        settings = get_optimal_settings()
        result = process_single_file(path, settings)
        
        if result['status'] == 'success':
            print(f"✅ Success! {result['output']} ({result['time']:.1f}s, {result['speed']} speed)")
        else:
            print(f"❌ Failed: {result['error']}")
    
    elif os.path.isdir(path):
        # Batch mode
        zip_files = find_zip_files(path)
        if not zip_files:
            print(f"No ZIP files found in {path}")
            sys.exit(1)
        
        print(f"Found {len(zip_files)} ZIP files")
        process_batch(zip_files, max_workers)
    
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)