# CSCI 576 Multimedia Project: Multimodal Video Segmentation

## Multimodal Segmentation of Long-Form Online Video into Content and Non-Content

**Course:** CSCI 576 - Multimedia System Design  
**Instructor:** Prof. Parag Havaldar  
**Demo Date:** May 6-8, 2026

---

## Project Overview

This project implements a multimodal video segmentation system that automatically segments long-form videos into:
- **Core Content** - The main material viewers want to watch
- **Non-Content** - Advertisements, intros, outros, transitions, silence, etc.

The system analyzes multiple modalities including:
- Visual features (color, brightness, scene changes, motion)
- Audio features (volume, spectral analysis, music detection)
- Temporal patterns and transitions

A custom video player is provided with an interactive timeline for navigation and skip functionality.

---

## Features

### Video Analysis
- **Visual Analysis**: Color histograms, brightness, contrast, edge density, motion estimation
- **Audio Analysis**: Volume levels, spectral centroid, zero-crossing rate, music vs speech detection
- **Scene Detection**: Automatic scene change detection using histogram comparison
- **Multimodal Fusion**: Combines visual and audio features for robust segmentation

### Video Player
- Interactive timeline showing segment types with color coding
- Click-to-seek on timeline
- Skip/Jump navigation between segments
- Auto-skip functionality for advertisements
- Segment list panel with clickable entries
- Video statistics display

### Segment Types Detected
| Type | Color | Description |
|------|-------|-------------|
| Core Content | Green | Main video content |
| Advertisement | Red | Ads, sponsorships |
| Intro | Blue | Video introductions |
| Outro | Purple | Video endings, credits |
| Transition | Amber | Transition screens, cards |
| Silence | Gray | Dead air, silence |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for audio extraction)

### Install Dependencies

```bash
# Clone/navigate to project directory
cd "D:\USC\Subjects\Multimedia System Design\Project"

# Install Python dependencies
pip install -r requirements.txt
```

### Install FFmpeg (Windows)
1. Download from https://ffmpeg.org/download.html
2. Extract and add to PATH
3. Or use: `choco install ffmpeg` (if using Chocolatey)

---

## Project Structure

```
Project/
├── main.py                 # Main entry point
├── run_demo.py            # Quick demo script
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── src/
│   ├── __init__.py
│   ├── video_analyzer.py  # Visual feature extraction
│   ├── audio_analyzer.py  # Audio feature extraction
│   ├── segmentation.py    # Multimodal segmentation
│   ├── video_player.py    # PyQt5 video player GUI
│   └── utils.py           # Utility functions
│
├── videos_with_ads/       # Test videos with embedded ads
│   ├── test_001.mp4
│   ├── test_002.mp4
│   └── ...
│
└── video_info/            # Ground truth JSON files
    ├── test_001.json
    ├── test_002.json
    └── ...
```

---

## Usage

### Quick Demo
```bash
python run_demo.py
```

### Launch Video Player
```bash
# Launch empty player
python main.py

# Open specific video
python main.py videos_with_ads/test_001.mp4
```

### Analyze Video
```bash
# Analyze single video
python main.py --analyze videos_with_ads/test_001.mp4

# Use ground truth segmentation
python main.py --analyze videos_with_ads/test_001.mp4 --ground-truth

# Batch analyze all videos
python main.py --batch videos_with_ads/
```

### List Available Videos
```bash
python main.py --list videos_with_ads/
```

### Command Line Options
```
usage: main.py [-h] [--analyze] [--batch DIR] [--output PATH]
               [--ground-truth] [--info-dir DIR] [--list DIR]
               [--segmentation PATH] [video]

Options:
  video                 Video file to open/analyze
  --analyze, -a        Analyze video instead of playing
  --batch DIR, -b      Batch analyze all videos in directory
  --output PATH, -o    Output path for analysis results
  --ground-truth, -g   Use ground truth from video_info
  --info-dir DIR       Directory containing video info JSONs
  --list DIR, -l       List videos in directory
  --segmentation PATH  Load existing segmentation JSON
```

---

## Video Player Controls

| Control | Action |
|---------|--------|
| Play/Pause Button | Toggle playback |
| Stop Button | Stop playback |
| Timeline Click | Seek to position |
| ◀ Prev | Jump to previous segment |
| Next ▶ | Jump to next segment |
| Skip Ad ⏭ | Skip current ad segment |
| Auto-skip ads | Automatically skip ads |
| Volume Slider | Adjust volume |
| Segment List | Click any segment to jump |

---

## Technical Details

### Visual Feature Extraction
```python
# Features extracted per frame:
- Color histogram (HSV, 32 bins per channel)
- Mean brightness
- Contrast (standard deviation)
- Edge density (Canny edge detection)
- Dominant colors (k-means clustering)
- Motion magnitude (optical flow)
```

### Audio Feature Extraction
```python
# Features extracted per segment:
- RMS energy (volume)
- Spectral centroid (brightness)
- Zero-crossing rate
- Spectral rolloff
- Spectral flatness (tonality)
- Music probability estimate
```

### Segmentation Algorithm
1. Extract visual features from video frames (sampled)
2. Extract audio features from audio track
3. Detect segment boundaries using:
   - Scene changes (histogram correlation)
   - Audio transitions (volume spikes)
   - Silence/music boundaries
4. Classify segments using multimodal features:
   - Scene change density
   - Music ratio
   - Volume levels
   - Position in video (start/end)
   - Segment duration
5. Merge adjacent segments of same type

---

## Ground Truth Data Format

The `video_info/` directory contains JSON files with ground truth:

```json
{
  "video_filename": "test_001.mp4",
  "output_duration_seconds": 1458.603,
  "num_ads_inserted": 3,
  "total_ads_duration_seconds": 178.724,
  "timeline_segments": [
    {
      "type": "video_content",
      "final_video_start_seconds": 0.0,
      "final_video_end_seconds": 106.159,
      "duration_seconds": 106.159
    },
    {
      "type": "ad",
      "ad_filename": "ads_009.mp4",
      "final_video_start_seconds": 106.159,
      "final_video_end_seconds": 224.395,
      "duration_seconds": 118.236
    }
  ]
}
```

---

## API Reference

### VideoAnalyzer
```python
from src.video_analyzer import VideoAnalyzer

analyzer = VideoAnalyzer("video.mp4")
results = analyzer.analyze_video(sample_rate=5)

# Access results
print(results['video_info'])
print(results['scene_changes'])
print(results['statistics'])
```

### AudioAnalyzer
```python
from src.audio_analyzer import AudioAnalyzer

analyzer = AudioAnalyzer("video.mp4")
results = analyzer.analyze_audio(segment_duration=1.0)

# Access results
print(results['statistics'])
print(results['silence_regions'])
print(results['music_regions'])
```

### Segmentation
```python
from src.segmentation import segment_video, SegmentationResult

# Analyze video
result = segment_video("video.mp4")

# Or use ground truth
result = segment_video("video.mp4", 
                       info_path="video_info/test_001.json",
                       use_ground_truth=True)

# Save/load results
result.save("output.json")
result = SegmentationResult.load("output.json")

# Access segments
for seg in result.segments:
    print(f"{seg.start_time:.1f}s - {seg.end_time:.1f}s: {seg.segment_type.value}")
```

### Video Player
```python
from src.video_player import run_player

# Launch player
run_player(video_path="video.mp4")

# With pre-loaded segmentation
run_player(video_path="video.mp4", 
           segmentation_path="output.json")
```

---

## Evaluation Metrics

The system can compute evaluation metrics comparing detected segments to ground truth:

```python
from src.utils import calculate_metrics

metrics = calculate_metrics(detected_segments, ground_truth_segments)
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"F1 Score: {metrics['f1']:.2%}")
print(f"IoU: {metrics['iou']:.2%}")
```

---

## Test Videos

The project includes 5 test videos with embedded advertisements:

| Video | Duration | Ads | Ad Duration |
|-------|----------|-----|-------------|
| test_001.mp4 | 24:18 | 3 | 178.7s |
| test_002.mp4 | 22:30 | 3 | 150.3s |
| test_003.mp4 | 30:26 | 3 | 186.0s |
| test_004.mp4 | 32:15 | 3 | 135.9s |
| test_005.mp4 | 23:39 | 3 | 105.2s |

---

## Troubleshooting

### FFmpeg not found
```
Error: FFmpeg not found
Solution: Install FFmpeg and add to PATH
```

### Video won't play
```
Possible causes:
1. Missing video codecs - install K-Lite Codec Pack
2. Corrupted video file
3. Unsupported format - convert to MP4
```

### PyQt5 display issues
```
Solution: Update graphics drivers or use software rendering
set QT_OPENGL=software
```

---

## Future Improvements

1. **Deep Learning Integration**: Use CNN/RNN models for better classification
2. **Speech Recognition**: Analyze transcripts for sponsor mentions
3. **Template Matching**: Detect repeated intro/outro sequences
4. **User Feedback Loop**: Learn from user skip behavior
5. **Real-time Processing**: Stream analysis for live content

---

## References

1. OpenCV Documentation: https://docs.opencv.org/
2. FFmpeg Documentation: https://ffmpeg.org/documentation.html
3. PyQt5 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt5/
4. SciPy Signal Processing: https://docs.scipy.org/doc/scipy/reference/signal.html

---

## License

This project is for educational purposes as part of CSCI 576 coursework at USC.

---

## Authors

CSCI 576 Multimedia System Design - Spring 2026
