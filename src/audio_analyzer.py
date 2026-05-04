"""
Audio Analyzer Module
Extracts audio features from video for content segmentation.

Features extracted:
- Volume levels (RMS energy)
- Frequency spectrum analysis
- Silence detection
- Music vs speech classification
- Audio tempo and rhythm
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import subprocess
import tempfile
import os
from scipy import signal
from scipy.io import wavfile
import warnings

warnings.filterwarnings('ignore')


@dataclass
class AudioSegment:
    """Represents an audio segment with its features."""
    start_time: float
    end_time: float
    avg_volume: float
    max_volume: float
    is_silence: bool
    is_music: bool
    spectral_centroid: float
    zero_crossing_rate: float


class AudioAnalyzer:
    """
    Analyzes audio track of a video to extract features for content segmentation.
    Uses audio cues to identify ads, intros, outros, and other non-content segments.
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0
        self.segments: List[AudioSegment] = []
        
        self._temp_audio_file = None
    
    def __del__(self):
        if self._temp_audio_file and os.path.exists(self._temp_audio_file):
            try:
                os.remove(self._temp_audio_file)
            except:
                pass
    
    def extract_audio(self) -> bool:
        """
        Extract audio from video file using ffmpeg.
        Returns True if successful.
        """
        try:
            fd, self._temp_audio_file = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            cmd = [
                'ffmpeg', '-y', '-i', self.video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '22050',  # Sample rate
                '-ac', '1',  # Mono
                self._temp_audio_file
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr.decode()}")
                return False
            
            self.sample_rate, self.audio_data = wavfile.read(self._temp_audio_file)
            self.audio_data = self.audio_data.astype(np.float32) / 32768.0
            self.duration = len(self.audio_data) / self.sample_rate
            
            return True
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False
    
    def calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) energy of audio chunk."""
        return float(np.sqrt(np.mean(audio_chunk ** 2)))
    
    def calculate_spectral_centroid(self, audio_chunk: np.ndarray) -> float:
        """
        Calculate spectral centroid (brightness of sound).
        Higher values indicate brighter/more treble sound.
        """
        if len(audio_chunk) < 256:
            return 0.0
        
        fft = np.fft.rfft(audio_chunk)
        magnitude = np.abs(fft)
        frequencies = np.fft.rfftfreq(len(audio_chunk), 1.0 / self.sample_rate)
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
        return float(centroid)
    
    def calculate_zero_crossing_rate(self, audio_chunk: np.ndarray) -> float:
        """
        Calculate zero-crossing rate.
        Higher ZCR often indicates speech or noise.
        """
        signs = np.sign(audio_chunk)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return float(crossings / len(audio_chunk))
    
    def calculate_spectral_rolloff(self, audio_chunk: np.ndarray, percentile: float = 0.85) -> float:
        """
        Calculate spectral rolloff frequency.
        Frequency below which percentile of spectrum energy is contained.
        """
        if len(audio_chunk) < 256:
            return 0.0
        
        fft = np.fft.rfft(audio_chunk)
        magnitude = np.abs(fft) ** 2
        frequencies = np.fft.rfftfreq(len(audio_chunk), 1.0 / self.sample_rate)
        
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] == 0:
            return 0.0
        
        rolloff_idx = np.searchsorted(cumsum, percentile * cumsum[-1])
        return float(frequencies[min(rolloff_idx, len(frequencies) - 1)])
    
    def calculate_spectral_flatness(self, audio_chunk: np.ndarray) -> float:
        """
        Calculate spectral flatness (tonality).
        Values close to 1 indicate noise, close to 0 indicate tonal content.
        """
        if len(audio_chunk) < 256:
            return 0.0
        
        fft = np.fft.rfft(audio_chunk)
        magnitude = np.abs(fft) + 1e-10
        
        geometric_mean = np.exp(np.mean(np.log(magnitude)))
        arithmetic_mean = np.mean(magnitude)
        
        return float(geometric_mean / arithmetic_mean)
    
    def detect_silence(self, audio_chunk: np.ndarray, threshold: float = 0.01) -> bool:
        """Detect if audio chunk is silence."""
        rms = self.calculate_rms(audio_chunk)
        return rms < threshold
    
    def estimate_music_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Estimate probability that audio chunk contains music.
        Uses spectral features to distinguish music from speech.
        """
        if len(audio_chunk) < 256:
            return 0.0
        
        zcr = self.calculate_zero_crossing_rate(audio_chunk)
        spectral_flatness = self.calculate_spectral_flatness(audio_chunk)
        spectral_rolloff = self.calculate_spectral_rolloff(audio_chunk)
        
        music_score = 0.0
        
        if zcr < 0.1:
            music_score += 0.3
        
        if spectral_flatness < 0.3:
            music_score += 0.3
        
        if spectral_rolloff > 3000:
            music_score += 0.4
        
        return music_score
    
    def analyze_audio(self, segment_duration: float = 1.0, progress_callback=None) -> Dict:
        """
        Analyze the entire audio track.
        
        Args:
            segment_duration: Duration of each analysis segment in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing all analysis results
        """
        if self.audio_data is None:
            if not self.extract_audio():
                return {'error': 'Failed to extract audio'}
        
        self.segments = []
        samples_per_segment = int(segment_duration * self.sample_rate)
        num_segments = len(self.audio_data) // samples_per_segment
        
        for i in range(num_segments):
            start_sample = i * samples_per_segment
            end_sample = start_sample + samples_per_segment
            chunk = self.audio_data[start_sample:end_sample]
            
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            rms = self.calculate_rms(chunk)
            max_amp = float(np.max(np.abs(chunk)))
            is_silence = self.detect_silence(chunk)
            music_prob = self.estimate_music_probability(chunk)
            spectral_centroid = self.calculate_spectral_centroid(chunk)
            zcr = self.calculate_zero_crossing_rate(chunk)
            
            segment = AudioSegment(
                start_time=start_time,
                end_time=end_time,
                avg_volume=rms,
                max_volume=max_amp,
                is_silence=is_silence,
                is_music=music_prob > 0.6,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=zcr
            )
            self.segments.append(segment)
            
            if progress_callback:
                progress_callback((i + 1) / num_segments)
        
        return self.get_analysis_results()
    
    def get_analysis_results(self) -> Dict:
        """Return all analysis results."""
        return {
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'segments': self.segments,
            'statistics': self._calculate_statistics(),
            'silence_regions': self._find_silence_regions(),
            'music_regions': self._find_music_regions()
        }
    
    def _calculate_statistics(self) -> Dict:
        """Calculate summary statistics."""
        if not self.segments:
            return {}
        
        volumes = [s.avg_volume for s in self.segments]
        centroids = [s.spectral_centroid for s in self.segments]
        
        return {
            'avg_volume': float(np.mean(volumes)),
            'std_volume': float(np.std(volumes)),
            'max_volume': float(np.max(volumes)),
            'min_volume': float(np.min(volumes)),
            'avg_spectral_centroid': float(np.mean(centroids)),
            'silence_ratio': sum(1 for s in self.segments if s.is_silence) / len(self.segments),
            'music_ratio': sum(1 for s in self.segments if s.is_music) / len(self.segments)
        }
    
    def _find_silence_regions(self, min_duration: float = 2.0) -> List[Tuple[float, float]]:
        """Find continuous silence regions."""
        regions = []
        start = None
        
        for seg in self.segments:
            if seg.is_silence:
                if start is None:
                    start = seg.start_time
            else:
                if start is not None:
                    duration = seg.start_time - start
                    if duration >= min_duration:
                        regions.append((start, seg.start_time))
                    start = None
        
        if start is not None:
            duration = self.duration - start
            if duration >= min_duration:
                regions.append((start, self.duration))
        
        return regions
    
    def _find_music_regions(self, min_duration: float = 5.0) -> List[Tuple[float, float]]:
        """Find continuous music regions."""
        regions = []
        start = None
        
        for seg in self.segments:
            if seg.is_music:
                if start is None:
                    start = seg.start_time
            else:
                if start is not None:
                    duration = seg.start_time - start
                    if duration >= min_duration:
                        regions.append((start, seg.start_time))
                    start = None
        
        if start is not None:
            duration = self.duration - start
            if duration >= min_duration:
                regions.append((start, self.duration))
        
        return regions
    
    def get_volume_at_time(self, timestamp: float) -> float:
        """Get volume level at a specific timestamp."""
        for seg in self.segments:
            if seg.start_time <= timestamp < seg.end_time:
                return seg.avg_volume
        return 0.0
    
    def get_volume_profile(self) -> Tuple[List[float], List[float]]:
        """Get volume profile as (timestamps, volumes)."""
        timestamps = [(s.start_time + s.end_time) / 2 for s in self.segments]
        volumes = [s.avg_volume for s in self.segments]
        return timestamps, volumes


def analyze_audio_from_video(video_path: str, segment_duration: float = 1.0) -> Dict:
    """
    Convenience function to analyze audio from a video file.
    
    Args:
        video_path: Path to the video file
        segment_duration: Duration of each analysis segment
        
    Returns:
        Analysis results dictionary
    """
    analyzer = AudioAnalyzer(video_path)
    return analyzer.analyze_audio(segment_duration=segment_duration)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results = analyze_audio_from_video(sys.argv[1])
        if 'error' not in results:
            print(f"Audio duration: {results['duration']:.2f}s")
            print(f"Average volume: {results['statistics']['avg_volume']:.4f}")
            print(f"Silence regions: {len(results['silence_regions'])}")
            print(f"Music regions: {len(results['music_regions'])}")
