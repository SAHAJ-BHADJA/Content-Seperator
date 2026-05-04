"""
Launch the OpenCV-based video player (more reliable on Windows).
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from opencv_player import run_opencv_player

if __name__ == "__main__":
    videos_dir = project_root / 'videos_with_ads'
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        videos = list(videos_dir.glob('*.mp4'))
        video_path = str(videos[0]) if videos else None
    
    print("=" * 50)
    print("CSCI 576 Video Segmentation Player (OpenCV)")
    print("=" * 50)
    if video_path:
        print(f"Loading: {os.path.basename(video_path)}")
    print()
    
    run_opencv_player(video_path)
