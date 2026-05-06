"""
Speech Transcript Analyzer
Uses speech recognition to detect ad/sponsorship keywords in video audio.
"""

import os
import subprocess
import tempfile
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


AD_KEYWORDS = [
    "sponsored by", "sponsor", "sponsorship",
    "brought to you by", "thanks to",
    "use code", "promo code", "discount code", "coupon code",
    "check out", "check the link", "link in description", "link below",
    "subscribe", "hit the bell", "notification", "like and subscribe",
    "patreon", "merchandise", "merch store",
    "audible", "squarespace", "nordvpn", "raid shadow", "skillshare",
    "brilliant.org", "curiositystream", "dollar shave",
    "don't forget to", "make sure to",
    "before we continue", "word from our sponsor",
    "this video is sponsored", "today's sponsor",
    "special thanks to", "shoutout to"
]

INTRO_KEYWORDS = [
    "welcome back", "what's up", "hey everyone", "hello everyone",
    "hi guys", "hey guys", "what's going on",
    "in this video", "today we", "today i",
    "let's get started", "let's begin", "let's dive",
    "intro", "introduction"
]

OUTRO_KEYWORDS = [
    "thanks for watching", "thank you for watching",
    "see you next", "see you in the next",
    "until next time", "bye", "goodbye", "peace out",
    "don't forget to subscribe", "leave a comment",
    "outro", "that's all for today", "that's it for today",
    "catch you later", "take care"
]


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    start_time: float
    end_time: float
    text: str
    confidence: float
    detected_type: Optional[str] = None
    keywords_found: List[str] = None
    
    def __post_init__(self):
        if self.keywords_found is None:
            self.keywords_found = []


class TranscriptAnalyzer:
    """Analyzes video speech to detect ads, intros, outros."""
    
    def __init__(self, video_path: str, model_size: str = "base"):
        self.video_path = video_path
        self.model_size = model_size
        self.model = None
        self.transcript_segments: List[TranscriptSegment] = []
        self.audio_file = None
        
    def _extract_audio(self) -> Optional[str]:
        """Extract audio from video for transcription."""
        try:
            fd, audio_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            cmd = [
                'ffmpeg', '-y', '-i', self.video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                audio_path
            ]
            
            result = subprocess.run(
                cmd, capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode == 0:
                self.audio_file = audio_path
                return audio_path
        except Exception as e:
            print(f"Audio extraction error: {e}")
        return None
    
    def transcribe(self, progress_callback=None) -> List[TranscriptSegment]:
        """Transcribe the video audio."""
        if not WHISPER_AVAILABLE:
            print("Whisper not available, using fallback")
            return self._fallback_analysis()
        
        if progress_callback:
            progress_callback(0.1, "Extracting audio...")
        
        audio_path = self._extract_audio()
        if not audio_path:
            return []
        
        try:
            if progress_callback:
                progress_callback(0.2, f"Loading Whisper model ({self.model_size})...")
            
            self.model = whisper.load_model(self.model_size)
            
            if progress_callback:
                progress_callback(0.3, "Transcribing audio...")
            
            result = self.model.transcribe(
                audio_path,
                language="en",
                verbose=False
            )
            
            if progress_callback:
                progress_callback(0.8, "Analyzing transcript...")
            
            for segment in result.get("segments", []):
                ts = TranscriptSegment(
                    start_time=segment["start"],
                    end_time=segment["end"],
                    text=segment["text"].strip(),
                    confidence=segment.get("no_speech_prob", 0)
                )
                self.transcript_segments.append(ts)
            
            self._analyze_keywords()
            
            if progress_callback:
                progress_callback(1.0, "Complete")
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return self._fallback_analysis()
        finally:
            if self.audio_file and os.path.exists(self.audio_file):
                try:
                    os.remove(self.audio_file)
                except:
                    pass
        
        return self.transcript_segments
    
    def _fallback_analysis(self) -> List[TranscriptSegment]:
        """Fallback when Whisper is not available."""
        return []
    
    def _analyze_keywords(self):
        """Analyze transcript for ad/intro/outro keywords."""
        for segment in self.transcript_segments:
            text_lower = segment.text.lower()
            
            ad_matches = [kw for kw in AD_KEYWORDS if kw in text_lower]
            if ad_matches:
                segment.detected_type = "ad"
                segment.keywords_found = ad_matches
                continue
            
            intro_matches = [kw for kw in INTRO_KEYWORDS if kw in text_lower]
            if intro_matches:
                segment.detected_type = "intro"
                segment.keywords_found = intro_matches
                continue
            
            outro_matches = [kw for kw in OUTRO_KEYWORDS if kw in text_lower]
            if outro_matches:
                segment.detected_type = "outro"
                segment.keywords_found = outro_matches
    
    def get_ad_segments(self) -> List[Tuple[float, float]]:
        """Get time ranges that likely contain ads."""
        ad_times = []
        current_start = None
        current_end = None
        
        for seg in self.transcript_segments:
            if seg.detected_type == "ad":
                if current_start is None:
                    current_start = seg.start_time
                current_end = seg.end_time
            else:
                if current_start is not None:
                    ad_times.append((current_start, current_end))
                    current_start = None
                    current_end = None
        
        if current_start is not None:
            ad_times.append((current_start, current_end))
        
        merged = []
        for start, end in ad_times:
            if merged and start - merged[-1][1] < 30:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        
        return merged
    
    def get_intro_time(self) -> Optional[float]:
        """Get estimated end time of intro."""
        for seg in self.transcript_segments[:20]:
            if seg.detected_type == "intro":
                return seg.end_time
        return None
    
    def get_outro_time(self) -> Optional[float]:
        """Get estimated start time of outro."""
        for seg in reversed(self.transcript_segments[-20:]):
            if seg.detected_type == "outro":
                return seg.start_time
        return None
    
    def get_full_transcript(self) -> str:
        """Get full transcript as text."""
        return " ".join([seg.text for seg in self.transcript_segments])
    
    def to_dict(self) -> Dict:
        """Export analysis results."""
        return {
            'video_path': self.video_path,
            'num_segments': len(self.transcript_segments),
            'ad_segments': self.get_ad_segments(),
            'intro_end': self.get_intro_time(),
            'outro_start': self.get_outro_time(),
            'transcript': [
                {
                    'start': s.start_time,
                    'end': s.end_time,
                    'text': s.text,
                    'type': s.detected_type,
                    'keywords': s.keywords_found
                }
                for s in self.transcript_segments
            ]
        }
    
    def save(self, output_path: str):
        """Save transcript analysis to JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def analyze_transcript(video_path: str, model_size: str = "base",
                       progress_callback=None) -> Dict:
    """Convenience function to analyze video transcript."""
    analyzer = TranscriptAnalyzer(video_path, model_size)
    analyzer.transcribe(progress_callback)
    return analyzer.to_dict()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_transcript(sys.argv[1])
        print(f"Transcript segments: {result['num_segments']}")
        print(f"Ad segments found: {len(result['ad_segments'])}")
        if result['intro_end']:
            print(f"Intro ends at: {result['intro_end']:.1f}s")
        if result['outro_start']:
            print(f"Outro starts at: {result['outro_start']:.1f}s")
