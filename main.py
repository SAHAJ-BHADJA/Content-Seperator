"""
CSCI 576 Multimedia Project - Multimodal Video Segmentation System
Main entry point for the application.

This system segments long-form videos into content and non-content segments
using multimodal analysis (visual, audio, motion features) and provides
an interactive video player for navigation.

Usage:
    python main.py                           # Launch GUI player
    python main.py <video_path>              # Open specific video
    python main.py --analyze <video_path>    # Analyze video and save results
    python main.py --batch <video_dir>       # Batch analyze all videos
"""

import sys
import os
import argparse
import json
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent


def setup_path():
    """Add src to Python path."""
    src_path = get_project_root() / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def launch_player(video_path=None, segmentation_path=None):
    """Launch the interactive video player."""
    setup_path()
    from src.video_player import run_player
    run_player(video_path, segmentation_path)


def analyze_video(video_path, output_path=None, use_ground_truth=False, info_dir=None):
    """
    Analyze a single video and save segmentation results.
    
    Args:
        video_path: Path to the video file
        output_path: Path for output JSON (default: same as video with .json extension)
        use_ground_truth: If True, use ground truth from info files
        info_dir: Directory containing video info JSON files
    """
    setup_path()
    from src.segmentation import segment_video, SegmentationResult
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return None
    
    info_path = None
    if info_dir:
        info_file = Path(info_dir) / f"{video_path.stem}.json"
        if info_file.exists():
            info_path = str(info_file)
    else:
        default_info = video_path.parent / 'video_info' / f"{video_path.stem}.json"
        if default_info.exists():
            info_path = str(default_info)
    
    print(f"\nAnalyzing: {video_path}")
    if info_path and use_ground_truth:
        print(f"Using ground truth: {info_path}")
    
    def progress_callback(value, message=""):
        bar_length = 40
        filled = int(bar_length * value)
        bar = '=' * filled + '-' * (bar_length - filled)
        status = message if message else f"{value:.0%}"
        print(f'\r[{bar}] {status}', end='', flush=True)
    
    result = segment_video(
        str(video_path),
        info_path=info_path,
        use_ground_truth=use_ground_truth,
        progress_callback=progress_callback if not use_ground_truth else None
    )
    
    print()
    
    if output_path is None:
        output_path = video_path.with_suffix('.segmentation.json')
    
    result.save(str(output_path))
    print(f"Saved segmentation to: {output_path}")
    
    print_summary(result)
    
    return result


def batch_analyze(video_dir, output_dir=None, use_ground_truth=False, info_dir=None):
    """
    Analyze all videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory for output files (default: same as video_dir)
        use_ground_truth: If True, use ground truth from info files
        info_dir: Directory containing video info JSON files
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"Error: Directory not found: {video_dir}")
        return
    
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}
    videos = [f for f in video_dir.iterdir() 
              if f.is_file() and f.suffix.lower() in video_extensions]
    
    if not videos:
        print(f"No video files found in: {video_dir}")
        return
    
    print(f"Found {len(videos)} video(s) to analyze")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing: {video_path.name}")
        
        if output_dir:
            out_path = output_dir / f"{video_path.stem}.segmentation.json"
        else:
            out_path = None
        
        result = analyze_video(
            str(video_path),
            output_path=str(out_path) if out_path else None,
            use_ground_truth=use_ground_truth,
            info_dir=info_dir
        )
        
        if result:
            results.append(result)
    
    print(f"\n{'='*60}")
    print(f"Batch analysis complete: {len(results)}/{len(videos)} videos processed")


def print_summary(result):
    """Print a summary of segmentation results."""
    print(f"\n{'='*50}")
    print(f"Video: {Path(result.video_path).name}")
    print(f"Duration: {format_time(result.duration)}")
    print(f"Segments: {len(result.segments)}")
    
    summary = result.to_dict()['summary']
    
    print(f"\nSegment breakdown:")
    for seg_type, duration in summary['type_durations'].items():
        count = summary['type_counts'].get(seg_type, 0)
        percentage = (duration / result.duration * 100) if result.duration > 0 else 0
        print(f"  {seg_type:15s}: {duration:7.1f}s ({percentage:5.1f}%) - {count} segment(s)")
    
    print(f"\nContent ratio: {summary['content_ratio']:.1%}")
    print(f"Ad ratio: {summary['ad_ratio']:.1%}")
    print(f"{'='*50}")


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def list_videos(video_dir):
    """List available videos and their segmentation status."""
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"Error: Directory not found: {video_dir}")
        return
    
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}
    videos = [f for f in video_dir.iterdir() 
              if f.is_file() and f.suffix.lower() in video_extensions]
    
    print(f"\nVideos in {video_dir}:")
    print(f"{'='*60}")
    
    for video in sorted(videos):
        seg_file = video.with_suffix('.segmentation.json')
        info_file = video.parent / 'video_info' / f"{video.stem}.json"
        
        status = []
        if seg_file.exists():
            status.append("segmented")
        if info_file.exists():
            status.append("has ground truth")
        
        status_str = f" [{', '.join(status)}]" if status else ""
        print(f"  {video.name}{status_str}")
    
    print(f"{'='*60}")
    print(f"Total: {len(videos)} video(s)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CSCI 576 Multimodal Video Segmentation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    Launch player GUI
  python main.py video.mp4                          Open video in player
  python main.py --analyze video.mp4                Analyze single video
  python main.py --batch ./videos_with_ads          Analyze all videos in directory
  python main.py --list ./videos_with_ads           List available videos
  python main.py --analyze video.mp4 --ground-truth Use ground truth segmentation
        """
    )
    
    parser.add_argument('video', nargs='?', help='Video file to open/analyze')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Analyze video instead of playing')
    parser.add_argument('--batch', '-b', metavar='DIR',
                       help='Batch analyze all videos in directory')
    parser.add_argument('--output', '-o', metavar='PATH',
                       help='Output path for analysis results')
    parser.add_argument('--ground-truth', '-g', action='store_true',
                       help='Use ground truth segmentation from video_info')
    parser.add_argument('--info-dir', metavar='DIR',
                       help='Directory containing video info JSON files')
    parser.add_argument('--list', '-l', metavar='DIR',
                       help='List videos in directory')
    parser.add_argument('--segmentation', '-s', metavar='PATH',
                       help='Load existing segmentation JSON for player')
    
    args = parser.parse_args()
    
    if args.list:
        list_videos(args.list)
        return
    
    if args.batch:
        batch_analyze(
            args.batch,
            output_dir=args.output,
            use_ground_truth=args.ground_truth,
            info_dir=args.info_dir
        )
        return
    
    if args.analyze and args.video:
        analyze_video(
            args.video,
            output_path=args.output,
            use_ground_truth=args.ground_truth,
            info_dir=args.info_dir
        )
        return
    
    launch_player(args.video, args.segmentation)


if __name__ == "__main__":
    main()
