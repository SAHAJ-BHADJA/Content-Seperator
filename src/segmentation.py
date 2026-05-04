"""
Multimodal Segmentation Module
Combines visual and audio features to segment video into content and non-content.

Segment types:
- core_content: Main video content
- ad: Advertisement/sponsorship
- intro: Video introduction
- outro: Video ending/credits
- transition: Transition screens, cards
- silence: Dead air, silence
- recap: Repeated content
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import os

try:
    from .video_analyzer import VideoAnalyzer, FrameFeatures, SceneChange
    from .audio_analyzer import AudioAnalyzer, AudioSegment
except ImportError:
    from video_analyzer import VideoAnalyzer, FrameFeatures, SceneChange
    from audio_analyzer import AudioAnalyzer, AudioSegment


class SegmentType(Enum):
    """Types of video segments."""
    CORE_CONTENT = "core_content"
    AD = "ad"
    INTRO = "intro"
    OUTRO = "outro"
    TRANSITION = "transition"
    SILENCE = "silence"
    RECAP = "recap"
    UNKNOWN = "unknown"


@dataclass
class VideoSegment:
    """Represents a classified video segment."""
    start_time: float
    end_time: float
    segment_type: SegmentType
    confidence: float
    features: Dict = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'start_formatted': self._format_time(self.start_time),
            'end_formatted': self._format_time(self.end_time),
            'duration': self.duration,
            'type': self.segment_type.value,
            'confidence': self.confidence,
            'features': self.features
        }
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class SegmentationResult:
    """Complete segmentation results for a video."""
    video_path: str
    duration: float
    segments: List[VideoSegment]
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'video_path': self.video_path,
            'duration': self.duration,
            'num_segments': len(self.segments),
            'segments': [s.to_dict() for s in self.segments],
            'summary': self._generate_summary(),
            'metadata': self.metadata
        }
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        type_durations = {}
        type_counts = {}
        
        for seg in self.segments:
            t = seg.segment_type.value
            type_durations[t] = type_durations.get(t, 0) + seg.duration
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            'type_durations': type_durations,
            'type_counts': type_counts,
            'content_ratio': type_durations.get('core_content', 0) / self.duration if self.duration > 0 else 0,
            'ad_ratio': type_durations.get('ad', 0) / self.duration if self.duration > 0 else 0
        }
    
    def save(self, output_path: str):
        """Save segmentation results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, input_path: str) -> 'SegmentationResult':
        """Load segmentation results from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        segments = []
        for s in data['segments']:
            segments.append(VideoSegment(
                start_time=s['start_time'],
                end_time=s['end_time'],
                segment_type=SegmentType(s['type']),
                confidence=s['confidence'],
                features=s.get('features', {})
            ))
        
        return cls(
            video_path=data['video_path'],
            duration=data['duration'],
            segments=segments,
            metadata=data.get('metadata', {})
        )


class MultimodalSegmenter:
    """
    Segments video into content and non-content using multimodal analysis.
    Combines visual features, audio features, and temporal patterns.
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_analyzer = VideoAnalyzer(video_path)
        self.audio_analyzer = AudioAnalyzer(video_path)
        
        self.video_features = None
        self.audio_features = None
        self.segments: List[VideoSegment] = []
        
        self.thresholds = {
            'scene_change_density': 0.5,
            'brightness_change': 50,
            'volume_spike': 0.3,
            'silence_threshold': 0.02,
            'music_threshold': 0.6,
            'min_segment_duration': 3.0,
            'ad_duration_range': (15, 180),
            'intro_max_duration': 60,
            'outro_max_from_end': 120
        }
    
    def analyze(self, video_sample_rate: int = 5, audio_segment_duration: float = 0.5,
                progress_callback=None) -> SegmentationResult:
        """
        Perform complete multimodal analysis and segmentation.
        
        Args:
            video_sample_rate: Analyze every Nth video frame
            audio_segment_duration: Duration of audio analysis segments
            progress_callback: Optional callback for progress updates
            
        Returns:
            SegmentationResult containing all detected segments
        """
        if progress_callback:
            progress_callback(0.0, "Analyzing video features...")
        self.video_features = self.video_analyzer.analyze_video(sample_rate=video_sample_rate)
        
        if progress_callback:
            progress_callback(0.4, "Analyzing audio features...")
        self.audio_features = self.audio_analyzer.analyze_audio(segment_duration=audio_segment_duration)
        
        if progress_callback:
            progress_callback(0.7, "Detecting segments...")
        self._detect_segments()
        
        if progress_callback:
            progress_callback(0.9, "Classifying segments...")
        self._classify_segments()
        
        self._merge_adjacent_segments()
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        return SegmentationResult(
            video_path=self.video_path,
            duration=self.video_analyzer.duration,
            segments=self.segments,
            metadata={
                'video_info': self.video_features['video_info'],
                'audio_duration': self.audio_features.get('duration', 0),
                'analysis_settings': {
                    'video_sample_rate': video_sample_rate,
                    'audio_segment_duration': audio_segment_duration
                }
            }
        )
    
    def _detect_segments(self):
        """Detect segment boundaries using scene changes and audio transitions."""
        boundaries = [0.0]
        
        for sc in self.video_features.get('scene_changes', []):
            if sc.confidence > 0.4:
                boundaries.append(sc.timestamp)
        
        audio_segments = self.audio_features.get('segments', [])
        for i in range(1, len(audio_segments)):
            prev_seg = audio_segments[i - 1]
            curr_seg = audio_segments[i]
            
            if abs(curr_seg.avg_volume - prev_seg.avg_volume) > self.thresholds['volume_spike']:
                boundaries.append(curr_seg.start_time)
            
            if prev_seg.is_silence != curr_seg.is_silence:
                boundaries.append(curr_seg.start_time)
            
            if prev_seg.is_music != curr_seg.is_music:
                boundaries.append(curr_seg.start_time)
        
        boundaries.append(self.video_analyzer.duration)
        boundaries = sorted(set(boundaries))
        
        merged_boundaries = [boundaries[0]]
        for b in boundaries[1:]:
            if b - merged_boundaries[-1] >= self.thresholds['min_segment_duration']:
                merged_boundaries.append(b)
            else:
                pass
        
        if merged_boundaries[-1] < self.video_analyzer.duration:
            merged_boundaries.append(self.video_analyzer.duration)
        
        self.segments = []
        for i in range(len(merged_boundaries) - 1):
            self.segments.append(VideoSegment(
                start_time=merged_boundaries[i],
                end_time=merged_boundaries[i + 1],
                segment_type=SegmentType.UNKNOWN,
                confidence=0.0
            ))
    
    def _classify_segments(self):
        """Classify each segment based on multimodal features."""
        for segment in self.segments:
            features = self._extract_segment_features(segment)
            segment.features = features
            
            segment_type, confidence = self._classify_single_segment(segment, features)
            segment.segment_type = segment_type
            segment.confidence = confidence
    
    def _extract_segment_features(self, segment: VideoSegment) -> Dict:
        """Extract combined features for a segment."""
        features = {}
        
        frame_features = [
            f for f in self.video_features.get('frame_features', [])
            if segment.start_time <= f.timestamp < segment.end_time
        ]
        
        if frame_features:
            features['avg_brightness'] = np.mean([f.brightness for f in frame_features])
            features['std_brightness'] = np.std([f.brightness for f in frame_features])
            features['avg_contrast'] = np.mean([f.contrast for f in frame_features])
            features['avg_edge_density'] = np.mean([f.edge_density for f in frame_features])
        
        scene_changes = [
            sc for sc in self.video_features.get('scene_changes', [])
            if segment.start_time <= sc.timestamp < segment.end_time
        ]
        features['scene_change_count'] = len(scene_changes)
        features['scene_change_density'] = len(scene_changes) / segment.duration if segment.duration > 0 else 0
        
        audio_segments = [
            s for s in self.audio_features.get('segments', [])
            if segment.start_time <= s.start_time < segment.end_time
        ]
        
        if audio_segments:
            features['avg_volume'] = np.mean([s.avg_volume for s in audio_segments])
            features['max_volume'] = max([s.max_volume for s in audio_segments])
            features['silence_ratio'] = sum(1 for s in audio_segments if s.is_silence) / len(audio_segments)
            features['music_ratio'] = sum(1 for s in audio_segments if s.is_music) / len(audio_segments)
            features['avg_spectral_centroid'] = np.mean([s.spectral_centroid for s in audio_segments])
        
        features['is_near_start'] = segment.start_time < self.thresholds['intro_max_duration']
        features['is_near_end'] = (self.video_analyzer.duration - segment.end_time) < self.thresholds['outro_max_from_end']
        features['duration'] = segment.duration
        
        return features
    
    def _classify_single_segment(self, segment: VideoSegment, features: Dict) -> Tuple[SegmentType, float]:
        """Classify a single segment based on its features."""
        scores = {
            SegmentType.CORE_CONTENT: 0.0,
            SegmentType.AD: 0.0,
            SegmentType.INTRO: 0.0,
            SegmentType.OUTRO: 0.0,
            SegmentType.TRANSITION: 0.0,
            SegmentType.SILENCE: 0.0
        }
        
        silence_ratio = features.get('silence_ratio', 0)
        if silence_ratio > 0.8:
            scores[SegmentType.SILENCE] += 0.9
        
        if features.get('scene_change_density', 0) > self.thresholds['scene_change_density']:
            scores[SegmentType.AD] += 0.3
            scores[SegmentType.TRANSITION] += 0.2
        
        music_ratio = features.get('music_ratio', 0)
        if music_ratio > 0.7:
            if features.get('is_near_start', False):
                scores[SegmentType.INTRO] += 0.5
            elif features.get('is_near_end', False):
                scores[SegmentType.OUTRO] += 0.5
            else:
                scores[SegmentType.AD] += 0.3
        
        duration = features.get('duration', 0)
        ad_min, ad_max = self.thresholds['ad_duration_range']
        if ad_min <= duration <= ad_max:
            scores[SegmentType.AD] += 0.2
        
        avg_volume = features.get('avg_volume', 0)
        overall_avg = self.audio_features.get('statistics', {}).get('avg_volume', 0.1)
        if avg_volume > overall_avg * 1.3:
            scores[SegmentType.AD] += 0.2
        
        if features.get('is_near_start', False) and duration < self.thresholds['intro_max_duration']:
            if music_ratio > 0.5 or features.get('std_brightness', 0) > 30:
                scores[SegmentType.INTRO] += 0.4
        
        if features.get('is_near_end', False):
            if music_ratio > 0.5 or silence_ratio > 0.3:
                scores[SegmentType.OUTRO] += 0.4
        
        edge_density = features.get('avg_edge_density', 0)
        if edge_density < 0.02 and features.get('std_brightness', float('inf')) < 10:
            scores[SegmentType.TRANSITION] += 0.5
        
        max_score = max(scores.values())
        if max_score < 0.3:
            return SegmentType.CORE_CONTENT, 0.7
        
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] + 0.3, 1.0)
        
        return best_type, confidence
    
    def _merge_adjacent_segments(self):
        """Merge adjacent segments of the same type."""
        if len(self.segments) < 2:
            return
        
        merged = [self.segments[0]]
        
        for segment in self.segments[1:]:
            prev = merged[-1]
            
            if (segment.segment_type == prev.segment_type and
                abs(segment.start_time - prev.end_time) < 0.5):
                
                merged_features = {}
                for key in set(prev.features.keys()) | set(segment.features.keys()):
                    if key in prev.features and key in segment.features:
                        if isinstance(prev.features[key], (int, float)):
                            total_duration = prev.duration + segment.duration
                            merged_features[key] = (
                                prev.features[key] * prev.duration +
                                segment.features[key] * segment.duration
                            ) / total_duration
                        else:
                            merged_features[key] = segment.features[key]
                    else:
                        merged_features[key] = prev.features.get(key, segment.features.get(key))
                
                prev.end_time = segment.end_time
                prev.confidence = (prev.confidence + segment.confidence) / 2
                prev.features = merged_features
            else:
                merged.append(segment)
        
        self.segments = merged


class GroundTruthSegmenter:
    """
    Creates segmentation from ground truth JSON files.
    Uses the provided video_info files which contain exact ad insertion points.
    """
    
    def __init__(self, video_path: str, info_path: str):
        self.video_path = video_path
        self.info_path = info_path
        
        with open(info_path, 'r') as f:
            self.info = json.load(f)
    
    def get_segmentation(self) -> SegmentationResult:
        """Create segmentation result from ground truth data."""
        segments = []
        
        for timeline_seg in self.info.get('timeline_segments', []):
            start_time = timeline_seg['final_video_start_seconds']
            end_time = timeline_seg['final_video_end_seconds']
            
            if timeline_seg['type'] == 'video_content':
                seg_type = SegmentType.CORE_CONTENT
            elif timeline_seg['type'] == 'ad':
                seg_type = SegmentType.AD
            else:
                seg_type = SegmentType.UNKNOWN
            
            segments.append(VideoSegment(
                start_time=start_time,
                end_time=end_time,
                segment_type=seg_type,
                confidence=1.0,
                features={
                    'source': 'ground_truth',
                    'ad_filename': timeline_seg.get('ad_filename', ''),
                    'ad_index': timeline_seg.get('ad_index', 0)
                }
            ))
        
        return SegmentationResult(
            video_path=self.video_path,
            duration=self.info.get('output_duration_seconds', 0),
            segments=segments,
            metadata={
                'source': 'ground_truth',
                'original_duration': self.info.get('original_video_duration_seconds', 0),
                'num_ads': self.info.get('num_ads_inserted', 0),
                'total_ads_duration': self.info.get('total_ads_duration_seconds', 0)
            }
        )


def segment_video(video_path: str, info_path: Optional[str] = None,
                  use_ground_truth: bool = False,
                  progress_callback=None) -> SegmentationResult:
    """
    Main function to segment a video.
    
    Args:
        video_path: Path to the video file
        info_path: Optional path to ground truth info JSON
        use_ground_truth: If True and info_path is provided, use ground truth
        progress_callback: Optional callback for progress updates
        
    Returns:
        SegmentationResult containing all detected segments
    """
    if use_ground_truth and info_path and os.path.exists(info_path):
        segmenter = GroundTruthSegmenter(video_path, info_path)
        return segmenter.get_segmentation()
    
    segmenter = MultimodalSegmenter(video_path)
    return segmenter.analyze(progress_callback=progress_callback)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        info_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        result = segment_video(video_path, info_path, use_ground_truth=bool(info_path))
        
        print(f"\nSegmentation Results for: {video_path}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Segments found: {len(result.segments)}")
        print("\nSegment breakdown:")
        
        for seg in result.segments:
            print(f"  [{seg.start_time:7.2f}s - {seg.end_time:7.2f}s] "
                  f"{seg.segment_type.value:15s} (confidence: {seg.confidence:.2f})")
