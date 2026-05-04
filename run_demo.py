"""
Quick Demo Script for CSCI 576 Multimodal Video Segmentation Project

This script demonstrates the system by:
1. Loading a test video with ground truth segmentation
2. Displaying the segmentation analysis
3. Launching the interactive player

Usage:
    python run_demo.py              # Run with first available test video
    python run_demo.py test_001     # Run with specific test video
"""

import sys
import os
from pathlib import Path


def main():
    project_root = Path(__file__).parent
    videos_dir = project_root / 'videos_with_ads'
    info_dir = project_root / 'video_info'
    
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print("=" * 60)
    print("CSCI 576 Multimedia Project - Video Segmentation Demo")
    print("=" * 60)
    
    from src.utils import check_dependencies
    
    deps = check_dependencies()
    missing = [name for name, available in deps.items() if not available]
    
    if missing:
        print("\nMissing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        if 'ffmpeg' in missing:
            print("  Also install ffmpeg: https://ffmpeg.org/download.html")
        return
    
    print("\nAll dependencies available!")
    
    video_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    if video_name:
        if not video_name.endswith('.mp4'):
            video_name += '.mp4'
        video_path = videos_dir / video_name
    else:
        videos = list(videos_dir.glob('*.mp4'))
        if not videos:
            print(f"\nNo videos found in {videos_dir}")
            print("Please add test videos to the videos_with_ads directory.")
            return
        video_path = videos[0]
    
    if not video_path.exists():
        print(f"\nVideo not found: {video_path}")
        print("\nAvailable videos:")
        for v in videos_dir.glob('*.mp4'):
            print(f"  - {v.name}")
        return
    
    info_path = info_dir / f"{video_path.stem}.json"
    
    print(f"\nVideo: {video_path.name}")
    print(f"Ground truth: {'Available' if info_path.exists() else 'Not found'}")
    
    from src.segmentation import GroundTruthSegmenter, SegmentType
    from src.utils import format_time
    
    if info_path.exists():
        print("\n" + "-" * 40)
        print("Ground Truth Segmentation:")
        print("-" * 40)
        
        segmenter = GroundTruthSegmenter(str(video_path), str(info_path))
        result = segmenter.get_segmentation()
        
        for seg in result.segments:
            seg_type = "CONTENT" if seg.segment_type == SegmentType.CORE_CONTENT else "AD"
            print(f"  [{format_time(seg.start_time)} - {format_time(seg.end_time)}] "
                  f"{seg_type:8s} ({seg.duration:.1f}s)")
        
        summary = result.to_dict()['summary']
        print("\n" + "-" * 40)
        print(f"Total duration: {format_time(result.duration)}")
        print(f"Content: {summary['type_durations'].get('core_content', 0):.1f}s "
              f"({summary['content_ratio']:.1%})")
        print(f"Ads: {summary['type_durations'].get('ad', 0):.1f}s "
              f"({summary['ad_ratio']:.1%})")
        print(f"Number of ads: {result.metadata.get('num_ads', 0)}")
    
    print("\n" + "=" * 60)
    print("Launching Video Player...")
    print("=" * 60)
    print("\nControls:")
    print("  - Play/Pause: Space or Play button")
    print("  - Seek: Click on timeline")
    print("  - Skip Ad: Click 'Skip Ad' button or use keyboard")
    print("  - Next/Prev Segment: Use navigation buttons")
    print("  - Auto-skip: Enable checkbox to auto-skip ads")
    print()
    
    from src.video_player import run_player
    run_player(str(video_path))


if __name__ == "__main__":
    main()
