"""
Video Analyzer Module
Extracts visual features from video frames for content segmentation.

Features extracted:
- Color histograms (HSV color space)
- Scene change detection
- Motion estimation between frames
- Frame brightness and contrast
- Edge density (visual complexity)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class FrameFeatures:
    """Features extracted from a single video frame."""
    frame_idx: int
    timestamp: float
    color_histogram: np.ndarray
    brightness: float
    contrast: float
    edge_density: float
    dominant_colors: List[Tuple[int, int, int]]


@dataclass
class SceneChange:
    """Represents a detected scene change."""
    frame_idx: int
    timestamp: float
    confidence: float
    change_type: str  # 'cut', 'fade', 'dissolve'


class VideoAnalyzer:
    """
    Analyzes video frames to extract visual features for content segmentation.
    Uses multiple visual cues to identify non-content segments like ads, intros, etc.
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.frame_features: List[FrameFeatures] = []
        self.scene_changes: List[SceneChange] = []
        self.motion_scores: List[float] = []
        
    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def get_video_info(self) -> Dict:
        """Return basic video information."""
        return {
            'path': self.video_path,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration': self.duration,
            'width': self.width,
            'height': self.height
        }
    
    def extract_color_histogram(self, frame: np.ndarray, bins: int = 32) -> np.ndarray:
        """
        Extract color histogram in HSV color space.
        HSV is more robust to lighting changes than RGB.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        histogram = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        histogram = histogram / (histogram.sum() + 1e-7)
        
        return histogram
    
    def calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate average brightness of frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    
    def calculate_contrast(self, frame: np.ndarray) -> float:
        """Calculate contrast (standard deviation of pixel values)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
    
    def calculate_edge_density(self, frame: np.ndarray) -> float:
        """
        Calculate edge density using Canny edge detection.
        Higher edge density = more visual complexity.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return float(edge_density)
    
    def get_dominant_colors(self, frame: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Extract k dominant colors using k-means clustering."""
        pixels = frame.reshape(-1, 3).astype(np.float32)
        
        sample_size = min(10000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sampled_pixels = pixels[indices]
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(sampled_pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        colors = [(int(c[2]), int(c[1]), int(c[0])) for c in centers]  # BGR to RGB
        return colors
    
    def calculate_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray, fast_mode: bool = True) -> float:
        """
        Calculate motion between two frames.
        Fast mode uses frame difference, slow mode uses optical flow.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        if fast_mode:
            diff = cv2.absdiff(prev_gray, curr_gray)
            return float(np.mean(diff))
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            return float(np.mean(magnitude))
    
    def detect_scene_change(self, prev_hist: np.ndarray, curr_hist: np.ndarray,
                           threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Detect scene change by comparing color histograms.
        Returns (is_scene_change, confidence).
        """
        correlation = cv2.compareHist(
            prev_hist.astype(np.float32),
            curr_hist.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        
        difference = 1 - correlation
        is_change = difference > threshold
        
        return is_change, difference
    
    def analyze_video(self, sample_rate: int = 1, progress_callback=None, fast_mode: bool = True) -> Dict:
        """
        Analyze the entire video and extract features.
        
        Args:
            sample_rate: Analyze every Nth frame (1 = all frames)
            progress_callback: Optional callback for progress updates
            fast_mode: If True, use faster but less accurate analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.frame_features = []
        self.scene_changes = []
        self.motion_scores = []
        
        prev_frame = None
        prev_histogram = None
        frame_idx = 0
        
        total_frames = self.frame_count // sample_rate
        pbar = tqdm(total=total_frames, desc="Analyzing video", disable=progress_callback is not None)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
            
            timestamp = frame_idx / self.fps
            
            if fast_mode:
                small_frame = cv2.resize(frame, (160, 90))
                histogram = self.extract_color_histogram(small_frame, bins=16)
                brightness = self.calculate_brightness(small_frame)
                contrast = self.calculate_contrast(small_frame)
                edge_density = 0.0
                dominant_colors = []
            else:
                histogram = self.extract_color_histogram(frame)
                brightness = self.calculate_brightness(frame)
                contrast = self.calculate_contrast(frame)
                edge_density = self.calculate_edge_density(frame)
                dominant_colors = self.get_dominant_colors(frame)
            
            features = FrameFeatures(
                frame_idx=frame_idx,
                timestamp=timestamp,
                color_histogram=histogram,
                brightness=brightness,
                contrast=contrast,
                edge_density=edge_density,
                dominant_colors=dominant_colors
            )
            self.frame_features.append(features)
            
            if prev_frame is not None:
                motion = self.calculate_motion(prev_frame, frame, fast_mode=fast_mode)
                self.motion_scores.append(motion)
                
                is_change, confidence = self.detect_scene_change(prev_histogram, histogram)
                if is_change:
                    change_type = 'cut' if confidence > 0.7 else 'fade'
                    self.scene_changes.append(SceneChange(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        confidence=confidence,
                        change_type=change_type
                    ))
            
            prev_frame = frame.copy()
            prev_histogram = histogram
            frame_idx += 1
            
            pbar.update(1)
            if progress_callback:
                progress_callback(frame_idx / self.frame_count)
        
        pbar.close()
        
        return self.get_analysis_results()
    
    def get_analysis_results(self) -> Dict:
        """Return all analysis results as a dictionary."""
        return {
            'video_info': self.get_video_info(),
            'frame_features': self.frame_features,
            'scene_changes': self.scene_changes,
            'motion_scores': self.motion_scores,
            'statistics': self._calculate_statistics()
        }
    
    def _calculate_statistics(self) -> Dict:
        """Calculate summary statistics from the analysis."""
        if not self.frame_features:
            return {}
        
        brightness_values = [f.brightness for f in self.frame_features]
        contrast_values = [f.contrast for f in self.frame_features]
        edge_values = [f.edge_density for f in self.frame_features]
        
        return {
            'avg_brightness': np.mean(brightness_values),
            'std_brightness': np.std(brightness_values),
            'avg_contrast': np.mean(contrast_values),
            'std_contrast': np.std(contrast_values),
            'avg_edge_density': np.mean(edge_values),
            'std_edge_density': np.std(edge_values),
            'num_scene_changes': len(self.scene_changes),
            'avg_motion': np.mean(self.motion_scores) if self.motion_scores else 0,
            'scene_change_rate': len(self.scene_changes) / self.duration if self.duration > 0 else 0
        }
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """Get a specific frame at the given timestamp."""
        frame_idx = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def get_thumbnail(self, timestamp: float, width: int = 160, height: int = 90) -> Optional[np.ndarray]:
        """Get a thumbnail image at the given timestamp."""
        frame = self.get_frame_at_time(timestamp)
        if frame is not None:
            return cv2.resize(frame, (width, height))
        return None


def analyze_video_file(video_path: str, sample_rate: int = 5) -> Dict:
    """
    Convenience function to analyze a video file.
    
    Args:
        video_path: Path to the video file
        sample_rate: Analyze every Nth frame
        
    Returns:
        Analysis results dictionary
    """
    analyzer = VideoAnalyzer(video_path)
    return analyzer.analyze_video(sample_rate=sample_rate)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results = analyze_video_file(sys.argv[1])
        print(f"Video duration: {results['video_info']['duration']:.2f}s")
        print(f"Scene changes detected: {len(results['scene_changes'])}")
        print(f"Average motion: {results['statistics']['avg_motion']:.4f}")
