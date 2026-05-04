"""
Utility functions for the Video Segmentation System.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


def format_time(seconds: float, include_ms: bool = False) -> str:
    """
    Format seconds to human readable time string.
    
    Args:
        seconds: Time in seconds
        include_ms: Include milliseconds in output
        
    Returns:
        Formatted time string (HH:MM:SS or HH:MM:SS.mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"


def parse_time(time_str: str) -> float:
    """
    Parse time string to seconds.
    
    Args:
        time_str: Time string in format HH:MM:SS or MM:SS or SS
        
    Returns:
        Time in seconds
    """
    parts = time_str.strip().split(':')
    
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        return float(parts[0])


def find_video_files(directory: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Find all video files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of video extensions (default: common video formats)
        
    Returns:
        List of Path objects for found video files
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
    
    directory = Path(directory)
    videos = []
    
    for ext in extensions:
        videos.extend(directory.glob(f'*{ext}'))
        videos.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(set(videos))


def load_video_info(info_path: str) -> Dict:
    """
    Load video info JSON file.
    
    Args:
        info_path: Path to video info JSON
        
    Returns:
        Dictionary with video info
    """
    with open(info_path, 'r') as f:
        return json.load(f)


def find_info_file(video_path: str, info_dir: Optional[str] = None) -> Optional[str]:
    """
    Find the corresponding info JSON file for a video.
    
    Args:
        video_path: Path to video file
        info_dir: Optional directory to search for info files
        
    Returns:
        Path to info file if found, None otherwise
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    
    search_paths = []
    
    if info_dir:
        search_paths.append(Path(info_dir) / f'{video_name}.json')
    
    search_paths.extend([
        video_path.parent / 'video_info' / f'{video_name}.json',
        video_path.parent.parent / 'video_info' / f'{video_name}.json',
        video_path.parent / f'{video_name}.json',
        video_path.with_suffix('.json'),
    ])
    
    for path in search_paths:
        if path.exists():
            return str(path)
    
    return None


def calculate_metrics(detected_segments: List[Dict], ground_truth_segments: List[Dict]) -> Dict:
    """
    Calculate evaluation metrics comparing detected segments to ground truth.
    
    Args:
        detected_segments: List of detected segment dictionaries
        ground_truth_segments: List of ground truth segment dictionaries
        
    Returns:
        Dictionary with precision, recall, F1 score, IoU
    """
    total_duration = max(
        max(s['end_time'] for s in detected_segments) if detected_segments else 0,
        max(s['end_time'] for s in ground_truth_segments) if ground_truth_segments else 0
    )
    
    if total_duration == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
    
    resolution = 0.1
    num_points = int(total_duration / resolution) + 1
    
    detected_mask = [False] * num_points
    gt_mask = [False] * num_points
    
    for seg in detected_segments:
        if seg.get('type') == 'ad':
            start_idx = int(seg['start_time'] / resolution)
            end_idx = int(seg['end_time'] / resolution)
            for i in range(start_idx, min(end_idx, num_points)):
                detected_mask[i] = True
    
    for seg in ground_truth_segments:
        if seg.get('type') == 'ad':
            start_idx = int(seg['final_video_start_seconds'] / resolution)
            end_idx = int(seg['final_video_end_seconds'] / resolution)
            for i in range(start_idx, min(end_idx, num_points)):
                gt_mask[i] = True
    
    true_positives = sum(1 for d, g in zip(detected_mask, gt_mask) if d and g)
    false_positives = sum(1 for d, g in zip(detected_mask, gt_mask) if d and not g)
    false_negatives = sum(1 for d, g in zip(detected_mask, gt_mask) if not d and g)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    intersection = true_positives
    union = true_positives + false_positives + false_negatives
    iou = intersection / union if union > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'true_positives': true_positives * resolution,
        'false_positives': false_positives * resolution,
        'false_negatives': false_negatives * resolution
    }


def generate_srt_chapters(segments: List[Dict], output_path: str):
    """
    Generate SRT chapter file from segments.
    
    Args:
        segments: List of segment dictionaries
        output_path: Path for output SRT file
    """
    with open(output_path, 'w') as f:
        for i, seg in enumerate(segments, 1):
            start = format_time(seg['start_time'], include_ms=True).replace('.', ',')
            end = format_time(seg['end_time'], include_ms=True).replace('.', ',')
            
            label = seg.get('type', 'unknown').replace('_', ' ').title()
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{label}\n\n")


def generate_chapter_markers(segments: List[Dict]) -> str:
    """
    Generate YouTube-style chapter markers.
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        String with chapter markers
    """
    lines = []
    
    for seg in segments:
        time = format_time(seg['start_time'])
        if time.startswith('00:'):
            time = time[3:]
        
        label = seg.get('type', 'unknown').replace('_', ' ').title()
        lines.append(f"{time} {label}")
    
    return '\n'.join(lines)


@dataclass
class VideoMetadata:
    """Container for video metadata."""
    path: str
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int
    
    @classmethod
    def from_file(cls, video_path: str) -> 'VideoMetadata':
        """Extract metadata from video file using ffprobe."""
        import subprocess
        import json
        
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            video_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            data = json.loads(result.stdout)
            
            video_stream = next(
                (s for s in data['streams'] if s['codec_type'] == 'video'),
                {}
            )
            
            fps_str = video_stream.get('r_frame_rate', '0/1')
            fps_parts = fps_str.split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_str)
            
            return cls(
                path=video_path,
                duration=float(data['format'].get('duration', 0)),
                width=int(video_stream.get('width', 0)),
                height=int(video_stream.get('height', 0)),
                fps=fps,
                codec=video_stream.get('codec_name', 'unknown'),
                bitrate=int(data['format'].get('bit_rate', 0))
            )
        except Exception as e:
            return cls(
                path=video_path,
                duration=0,
                width=0,
                height=0,
                fps=0,
                codec='unknown',
                bitrate=0
            )


def ensure_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH."""
    import subprocess
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available."""
    results = {
        'ffmpeg': ensure_ffmpeg_available(),
    }
    
    try:
        import cv2
        results['opencv'] = True
    except ImportError:
        results['opencv'] = False
    
    try:
        import numpy
        results['numpy'] = True
    except ImportError:
        results['numpy'] = False
    
    try:
        from scipy import signal
        results['scipy'] = True
    except ImportError:
        results['scipy'] = False
    
    try:
        from PyQt5.QtWidgets import QApplication
        results['pyqt5'] = True
    except ImportError:
        results['pyqt5'] = False
    
    return results
