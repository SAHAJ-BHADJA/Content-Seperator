"""
Custom Video Player with Segment Timeline
Interactive video player with visual content map and navigation controls.

Features:
- Video playback with synchronized audio
- Interactive timeline showing segment types
- Click to seek on timeline
- Skip/jump to next content segment
- Auto-skip non-content option
- Segment type legend and statistics
"""

import sys
import os
from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QStyle, QSizePolicy, QFrame,
    QFileDialog, QMessageBox, QCheckBox, QComboBox, QGroupBox,
    QScrollArea, QToolTip, QStatusBar, QSplitter, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QUrl, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
    QPalette, QMouseEvent, QFontMetrics
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

from .segmentation import SegmentationResult, VideoSegment, SegmentType


SEGMENT_COLORS = {
    SegmentType.CORE_CONTENT: QColor(76, 175, 80),     # Green
    SegmentType.AD: QColor(244, 67, 54),               # Red
    SegmentType.INTRO: QColor(33, 150, 243),           # Blue
    SegmentType.OUTRO: QColor(156, 39, 176),           # Purple
    SegmentType.TRANSITION: QColor(255, 193, 7),       # Amber
    SegmentType.SILENCE: QColor(158, 158, 158),        # Gray
    SegmentType.RECAP: QColor(255, 152, 0),            # Orange
    SegmentType.UNKNOWN: QColor(96, 125, 139)          # Blue Gray
}

SEGMENT_LABELS = {
    SegmentType.CORE_CONTENT: "Content",
    SegmentType.AD: "Advertisement",
    SegmentType.INTRO: "Intro",
    SegmentType.OUTRO: "Outro",
    SegmentType.TRANSITION: "Transition",
    SegmentType.SILENCE: "Silence",
    SegmentType.RECAP: "Recap",
    SegmentType.UNKNOWN: "Unknown"
}


class SegmentTimeline(QWidget):
    """
    Interactive timeline widget showing video segments.
    Supports clicking to seek and hover tooltips.
    """
    
    seekRequested = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments: List[VideoSegment] = []
        self.duration = 0.0
        self.current_position = 0.0
        self.hover_position = -1
        
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        
    def set_segments(self, segments: List[VideoSegment], duration: float):
        """Set the segments to display."""
        self.segments = segments
        self.duration = duration
        self.update()
    
    def set_position(self, position: float):
        """Update the current playback position."""
        self.current_position = position
        self.update()
    
    def paintEvent(self, event):
        """Draw the timeline with segments."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width() - 20
        height = self.height() - 30
        x_offset = 10
        y_offset = 10
        
        painter.fillRect(x_offset, y_offset, width, height, QColor(45, 45, 45))
        
        if self.duration > 0 and self.segments:
            for segment in self.segments:
                x = x_offset + int((segment.start_time / self.duration) * width)
                w = max(1, int((segment.duration / self.duration) * width))
                
                color = SEGMENT_COLORS.get(segment.segment_type, QColor(128, 128, 128))
                
                gradient = QLinearGradient(x, y_offset, x, y_offset + height)
                gradient.setColorAt(0, color.lighter(120))
                gradient.setColorAt(1, color.darker(110))
                
                painter.fillRect(x, y_offset, w, height, gradient)
                
                painter.setPen(QPen(color.darker(150), 1))
                painter.drawRect(x, y_offset, w, height)
        
        if self.duration > 0:
            pos_x = x_offset + int((self.current_position / self.duration) * width)
            
            painter.setPen(QPen(Qt.white, 2))
            painter.drawLine(pos_x, y_offset - 3, pos_x, y_offset + height + 3)
            
            painter.setBrush(QBrush(Qt.white))
            painter.setPen(Qt.NoPen)
            triangle_size = 6
            triangle = [
                QPoint(pos_x, y_offset - 3),
                QPoint(pos_x - triangle_size, y_offset - 3 - triangle_size),
                QPoint(pos_x + triangle_size, y_offset - 3 - triangle_size)
            ]
            painter.drawPolygon(*triangle)
        
        painter.setPen(QPen(Qt.white, 1))
        painter.setFont(QFont('Arial', 8))
        
        num_markers = 10
        for i in range(num_markers + 1):
            x = x_offset + int((i / num_markers) * width)
            time = (i / num_markers) * self.duration
            
            painter.drawLine(x, y_offset + height, x, y_offset + height + 3)
            
            time_str = self._format_time(time)
            fm = QFontMetrics(painter.font())
            text_width = fm.horizontalAdvance(time_str)
            painter.drawText(x - text_width // 2, y_offset + height + 15, time_str)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle click to seek."""
        if event.button() == Qt.LeftButton:
            position = self._get_time_at_x(event.x())
            if 0 <= position <= self.duration:
                self.seekRequested.emit(position)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Show tooltip on hover."""
        position = self._get_time_at_x(event.x())
        if 0 <= position <= self.duration:
            segment = self._get_segment_at_time(position)
            if segment:
                tooltip = (
                    f"{SEGMENT_LABELS[segment.segment_type]}\n"
                    f"Time: {self._format_time(position)}\n"
                    f"Duration: {segment.duration:.1f}s\n"
                    f"Confidence: {segment.confidence:.0%}"
                )
                QToolTip.showText(event.globalPos(), tooltip)
    
    def _get_time_at_x(self, x: int) -> float:
        """Convert x coordinate to time."""
        width = self.width() - 20
        x_offset = 10
        ratio = (x - x_offset) / width
        return ratio * self.duration
    
    def _get_segment_at_time(self, time: float) -> Optional[VideoSegment]:
        """Get the segment at the given time."""
        for segment in self.segments:
            if segment.start_time <= time < segment.end_time:
                return segment
        return None
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


class SegmentLegend(QWidget):
    """Widget showing the legend for segment types."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.segment_stats: Dict[SegmentType, float] = {}
        self.setMinimumHeight(30)
        self.setMaximumHeight(40)
    
    def set_stats(self, stats: Dict[SegmentType, float]):
        """Set segment duration statistics."""
        self.segment_stats = stats
        self.update()
    
    def paintEvent(self, event):
        """Draw the legend."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        x = 10
        y = 10
        box_size = 15
        spacing = 15
        
        painter.setFont(QFont('Arial', 9))
        fm = QFontMetrics(painter.font())
        
        for seg_type, color in SEGMENT_COLORS.items():
            if seg_type in self.segment_stats or seg_type in [SegmentType.CORE_CONTENT, SegmentType.AD]:
                painter.fillRect(x, y, box_size, box_size, color)
                painter.setPen(QPen(color.darker(150), 1))
                painter.drawRect(x, y, box_size, box_size)
                
                label = SEGMENT_LABELS[seg_type]
                duration = self.segment_stats.get(seg_type, 0)
                if duration > 0:
                    label += f" ({duration:.0f}s)"
                
                painter.setPen(Qt.white)
                painter.drawText(x + box_size + 5, y + box_size - 3, label)
                
                x += box_size + 5 + fm.horizontalAdvance(label) + spacing
                
                if x > self.width() - 150:
                    x = 10
                    y += box_size + 5


class SegmentListWidget(QWidget):
    """Widget showing list of all segments."""
    
    seekRequested = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments: List[VideoSegment] = []
        self.current_segment_idx = -1
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        self.segment_buttons: List[QPushButton] = []
    
    def set_segments(self, segments: List[VideoSegment]):
        """Set the segments to display."""
        for btn in self.segment_buttons:
            btn.deleteLater()
        self.segment_buttons.clear()
        
        self.segments = segments
        
        for i, seg in enumerate(segments):
            color = SEGMENT_COLORS.get(seg.segment_type, QColor(128, 128, 128))
            label = SEGMENT_LABELS[seg.segment_type]
            
            btn_text = f"{self._format_time(seg.start_time)} - {label} ({seg.duration:.1f}s)"
            btn = QPushButton(btn_text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color.name()};
                    color: white;
                    border: none;
                    padding: 5px;
                    text-align: left;
                    border-radius: 3px;
                }}
                QPushButton:hover {{
                    background-color: {color.lighter(120).name()};
                }}
            """)
            btn.clicked.connect(lambda checked, t=seg.start_time: self.seekRequested.emit(t))
            
            self.layout().addWidget(btn)
            self.segment_buttons.append(btn)
        
        self.layout().addStretch()
    
    def highlight_segment(self, time: float):
        """Highlight the segment at the current time."""
        for i, seg in enumerate(self.segments):
            if seg.start_time <= time < seg.end_time:
                if i != self.current_segment_idx:
                    self.current_segment_idx = i
                    self._update_highlights()
                break
    
    def _update_highlights(self):
        """Update button highlighting."""
        for i, btn in enumerate(self.segment_buttons):
            if i < len(self.segments):
                seg = self.segments[i]
                color = SEGMENT_COLORS.get(seg.segment_type, QColor(128, 128, 128))
                
                if i == self.current_segment_idx:
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {color.name()};
                            color: white;
                            border: 3px solid white;
                            padding: 5px;
                            text-align: left;
                            border-radius: 3px;
                        }}
                    """)
                else:
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {color.name()};
                            color: white;
                            border: none;
                            padding: 5px;
                            text-align: left;
                            border-radius: 3px;
                        }}
                        QPushButton:hover {{
                            background-color: {color.lighter(120).name()};
                        }}
                    """)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


class VideoPlayer(QMainWindow):
    """
    Main video player window with segment timeline and navigation.
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("CSCI 576 - Multimodal Video Segmentation Player")
        self.setGeometry(100, 100, 1280, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QCheckBox {
                color: white;
            }
            QComboBox {
                background-color: #3d3d3d;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #3d3d3d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QGroupBox {
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
            }
        """)
        
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.segmentation_result: Optional[SegmentationResult] = None
        self.auto_skip_enabled = False
        self.skip_types = {SegmentType.AD}
        
        self._setup_ui()
        self._connect_signals()
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_position)
        self.update_timer.start(100)
    
    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(800, 450)
        self.video_widget.setStyleSheet("background-color: black;")
        left_layout.addWidget(self.video_widget)
        
        self.timeline = SegmentTimeline()
        left_layout.addWidget(self.timeline)
        
        self.legend = SegmentLegend()
        left_layout.addWidget(self.legend)
        
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setFixedSize(40, 40)
        controls_layout.addWidget(self.play_button)
        
        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setFixedSize(40, 40)
        controls_layout.addWidget(self.stop_button)
        
        self.prev_segment_btn = QPushButton("◀ Prev")
        self.prev_segment_btn.setToolTip("Jump to previous segment")
        controls_layout.addWidget(self.prev_segment_btn)
        
        self.next_segment_btn = QPushButton("Next ▶")
        self.next_segment_btn.setToolTip("Jump to next segment")
        controls_layout.addWidget(self.next_segment_btn)
        
        self.skip_ad_btn = QPushButton("Skip Ad ⏭")
        self.skip_ad_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e53935;
            }
        """)
        self.skip_ad_btn.setToolTip("Skip to end of current ad")
        controls_layout.addWidget(self.skip_ad_btn)
        
        controls_layout.addSpacing(20)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(120)
        controls_layout.addWidget(self.time_label)
        
        controls_layout.addSpacing(20)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.setMaximumWidth(100)
        controls_layout.addWidget(QLabel("Volume:"))
        controls_layout.addWidget(self.volume_slider)
        
        controls_layout.addStretch()
        
        self.auto_skip_checkbox = QCheckBox("Auto-skip ads")
        self.auto_skip_checkbox.setToolTip("Automatically skip advertisement segments")
        controls_layout.addWidget(self.auto_skip_checkbox)
        
        left_layout.addLayout(controls_layout)
        
        main_layout.addWidget(left_panel, stretch=3)
        
        right_panel = QWidget()
        right_panel.setMaximumWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        load_group = QGroupBox("Load Video")
        load_layout = QVBoxLayout(load_group)
        
        self.load_video_btn = QPushButton("Open Video File")
        load_layout.addWidget(self.load_video_btn)
        
        self.load_segmentation_btn = QPushButton("Load Segmentation")
        load_layout.addWidget(self.load_segmentation_btn)
        
        self.analyze_btn = QPushButton("Analyze Video")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #43A047;
            }
        """)
        load_layout.addWidget(self.analyze_btn)
        
        right_layout.addWidget(load_group)
        
        stats_group = QGroupBox("Video Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("No video loaded")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        right_layout.addWidget(stats_group)
        
        segments_group = QGroupBox("Segments")
        segments_layout = QVBoxLayout(segments_group)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        self.segment_list = SegmentListWidget()
        scroll_area.setWidget(self.segment_list)
        
        segments_layout.addWidget(scroll_area)
        
        right_layout.addWidget(segments_group, stretch=1)
        
        main_layout.addWidget(right_panel, stretch=1)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        self.media_player.setVideoOutput(self.video_widget)
    
    def _connect_signals(self):
        """Connect UI signals to slots."""
        self.play_button.clicked.connect(self._toggle_play)
        self.stop_button.clicked.connect(self._stop)
        self.prev_segment_btn.clicked.connect(self._prev_segment)
        self.next_segment_btn.clicked.connect(self._next_segment)
        self.skip_ad_btn.clicked.connect(self._skip_current_ad)
        
        self.volume_slider.valueChanged.connect(self._set_volume)
        self.auto_skip_checkbox.stateChanged.connect(self._toggle_auto_skip)
        
        self.load_video_btn.clicked.connect(self._load_video)
        self.load_segmentation_btn.clicked.connect(self._load_segmentation)
        self.analyze_btn.clicked.connect(self._analyze_video)
        
        self.timeline.seekRequested.connect(self._seek)
        self.segment_list.seekRequested.connect(self._seek)
        
        self.media_player.stateChanged.connect(self._media_state_changed)
        self.media_player.durationChanged.connect(self._duration_changed)
        self.media_player.error.connect(self._handle_error)
    
    def _toggle_play(self):
        """Toggle play/pause."""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
    
    def _stop(self):
        """Stop playback."""
        self.media_player.stop()
    
    def _seek(self, position: float):
        """Seek to position in seconds."""
        self.media_player.setPosition(int(position * 1000))
    
    def _set_volume(self, volume: int):
        """Set volume level."""
        self.media_player.setVolume(volume)
    
    def _toggle_auto_skip(self, state: int):
        """Toggle auto-skip feature."""
        self.auto_skip_enabled = state == Qt.Checked
    
    def _prev_segment(self):
        """Jump to previous segment."""
        if not self.segmentation_result:
            return
        
        current_time = self.media_player.position() / 1000.0
        
        for seg in reversed(self.segmentation_result.segments):
            if seg.end_time < current_time - 0.5:
                self._seek(seg.start_time)
                return
        
        self._seek(0)
    
    def _next_segment(self):
        """Jump to next segment."""
        if not self.segmentation_result:
            return
        
        current_time = self.media_player.position() / 1000.0
        
        for seg in self.segmentation_result.segments:
            if seg.start_time > current_time + 0.5:
                self._seek(seg.start_time)
                return
    
    def _skip_current_ad(self):
        """Skip to the end of current ad segment."""
        if not self.segmentation_result:
            return
        
        current_time = self.media_player.position() / 1000.0
        
        for seg in self.segmentation_result.segments:
            if seg.start_time <= current_time < seg.end_time:
                if seg.segment_type in [SegmentType.AD, SegmentType.INTRO, 
                                        SegmentType.OUTRO, SegmentType.TRANSITION]:
                    self._seek(seg.end_time)
                    self.statusBar.showMessage(f"Skipped {SEGMENT_LABELS[seg.segment_type]}")
                return
    
    def _update_position(self):
        """Update UI with current position."""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            position = self.media_player.position() / 1000.0
            duration = self.media_player.duration() / 1000.0
            
            self.timeline.set_position(position)
            self.time_label.setText(
                f"{self._format_time(position)} / {self._format_time(duration)}"
            )
            
            self.segment_list.highlight_segment(position)
            
            if self.auto_skip_enabled and self.segmentation_result:
                for seg in self.segmentation_result.segments:
                    if seg.start_time <= position < seg.start_time + 0.5:
                        if seg.segment_type in self.skip_types:
                            self._seek(seg.end_time)
                            self.statusBar.showMessage(f"Auto-skipped {SEGMENT_LABELS[seg.segment_type]}")
                            break
    
    def _media_state_changed(self, state: QMediaPlayer.State):
        """Handle media state changes."""
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
    
    def _duration_changed(self, duration: int):
        """Handle duration change."""
        duration_sec = duration / 1000.0
        self.time_label.setText(f"00:00 / {self._format_time(duration_sec)}")
    
    def _handle_error(self):
        """Handle media player errors."""
        error_msg = self.media_player.errorString()
        error_code = self.media_player.error()
        full_msg = f"Error ({error_code}): {error_msg}" if error_msg else f"Error code: {error_code}"
        self.statusBar.showMessage(full_msg)
    
    def _check_media_status(self, path: str):
        """Check if media loaded successfully."""
        status = self.media_player.mediaStatus()
        if status == QMediaPlayer.InvalidMedia:
            self.statusBar.showMessage(f"Cannot play: {os.path.basename(path)} - Try installing K-Lite Codec Pack")
        elif status == QMediaPlayer.NoMedia:
            self.statusBar.showMessage(f"No media loaded")
        elif status in [QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia]:
            self.statusBar.showMessage(f"Ready: {os.path.basename(path)}")
    
    def _load_video(self):
        """Open file dialog to load video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, video_path: str):
        """Load a video file."""
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "Error", f"Video file not found: {video_path}")
            return
        
        abs_path = os.path.abspath(video_path).replace('\\', '/')
        url = QUrl.fromLocalFile(abs_path)
        
        self.media_player.setMedia(QMediaContent(url))
        self.media_player.setVolume(self.volume_slider.value())
        
        QTimer.singleShot(500, lambda: self._check_media_status(abs_path))
        
        self.setWindowTitle(f"CSCI 576 Player - {os.path.basename(video_path)}")
        self.statusBar.showMessage(f"Loaded: {video_path}")
        
        info_path = self._find_info_file(video_path)
        if info_path:
            self._load_ground_truth(video_path, info_path)
    
    def _find_info_file(self, video_path: str) -> Optional[str]:
        """Find corresponding info JSON file."""
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        possible_paths = [
            os.path.join(video_dir, '..', 'video_info', f'{video_name}.json'),
            os.path.join(video_dir, 'video_info', f'{video_name}.json'),
            os.path.join(video_dir, f'{video_name}.json'),
        ]
        
        for path in possible_paths:
            normalized = os.path.normpath(path)
            if os.path.exists(normalized):
                return normalized
        
        return None
    
    def _load_ground_truth(self, video_path: str, info_path: str):
        """Load ground truth segmentation from info file."""
        try:
            from .segmentation import GroundTruthSegmenter
            
            segmenter = GroundTruthSegmenter(video_path, info_path)
            result = segmenter.get_segmentation()
            self.set_segmentation(result)
            
            self.statusBar.showMessage(f"Loaded ground truth segmentation from {info_path}")
        except Exception as e:
            self.statusBar.showMessage(f"Could not load ground truth: {e}")
    
    def _load_segmentation(self):
        """Load segmentation from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Segmentation", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                result = SegmentationResult.load(file_path)
                self.set_segmentation(result)
                self.statusBar.showMessage(f"Loaded segmentation: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load segmentation: {e}")
    
    def _analyze_video(self):
        """Analyze current video to detect segments."""
        if self.media_player.media().isNull():
            QMessageBox.warning(self, "Error", "Please load a video first")
            return
        
        video_path = self.media_player.media().canonicalUrl().toLocalFile()
        
        progress = QProgressBar()
        progress.setRange(0, 100)
        self.statusBar.addWidget(progress)
        
        self.statusBar.showMessage("Analyzing video...")
        QApplication.processEvents()
        
        try:
            from .segmentation import MultimodalSegmenter
            
            def update_progress(value, message=""):
                progress.setValue(int(value * 100))
                if message:
                    self.statusBar.showMessage(message)
                QApplication.processEvents()
            
            segmenter = MultimodalSegmenter(video_path)
            result = segmenter.analyze(progress_callback=update_progress)
            
            self.set_segmentation(result)
            self.statusBar.showMessage("Analysis complete")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Analysis failed: {e}")
            self.statusBar.showMessage(f"Analysis failed: {e}")
        
        finally:
            self.statusBar.removeWidget(progress)
    
    def set_segmentation(self, result: SegmentationResult):
        """Set the segmentation result and update UI."""
        self.segmentation_result = result
        
        self.timeline.set_segments(result.segments, result.duration)
        
        type_durations = {}
        for seg in result.segments:
            t = seg.segment_type
            type_durations[t] = type_durations.get(t, 0) + seg.duration
        self.legend.set_stats(type_durations)
        
        self.segment_list.set_segments(result.segments)
        
        summary = result.to_dict()['summary']
        stats_text = (
            f"Duration: {self._format_time(result.duration)}\n"
            f"Segments: {len(result.segments)}\n"
            f"Content: {summary['content_ratio']:.1%}\n"
            f"Ads: {summary['ad_ratio']:.1%}\n"
        )
        
        if result.metadata.get('num_ads'):
            stats_text += f"Ads inserted: {result.metadata['num_ads']}\n"
            stats_text += f"Total ad time: {result.metadata.get('total_ads_duration', 0):.1f}s"
        
        self.stats_label.setText(stats_text)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def closeEvent(self, event):
        """Clean up on close."""
        self.media_player.stop()
        super().closeEvent(event)


def run_player(video_path: Optional[str] = None, segmentation_path: Optional[str] = None):
    """
    Run the video player application.
    
    Args:
        video_path: Optional path to video file to load
        segmentation_path: Optional path to segmentation JSON to load
    """
    app = QApplication(sys.argv)
    
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    player = VideoPlayer()
    player.show()
    
    if video_path:
        player.load_video(video_path)
    
    if segmentation_path:
        try:
            result = SegmentationResult.load(segmentation_path)
            player.set_segmentation(result)
        except Exception as e:
            print(f"Error loading segmentation: {e}")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    seg = sys.argv[2] if len(sys.argv) > 2 else None
    run_player(video, seg)
