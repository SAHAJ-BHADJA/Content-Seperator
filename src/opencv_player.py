"""
OpenCV-based Video Player with Segment Timeline
Alternative player using OpenCV for reliable video playback on Windows.
"""

import sys
import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QStyle, QSizePolicy, QFrame,
    QFileDialog, QMessageBox, QCheckBox, QGroupBox,
    QScrollArea, QToolTip, QStatusBar, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
    QPalette, QMouseEvent, QFontMetrics, QImage, QPixmap
)

try:
    from .segmentation import SegmentationResult, VideoSegment, SegmentType
except ImportError:
    from segmentation import SegmentationResult, VideoSegment, SegmentType


SEGMENT_COLORS = {
    SegmentType.CORE_CONTENT: QColor(76, 175, 80),
    SegmentType.AD: QColor(244, 67, 54),
    SegmentType.INTRO: QColor(33, 150, 243),
    SegmentType.OUTRO: QColor(156, 39, 176),
    SegmentType.TRANSITION: QColor(255, 193, 7),
    SegmentType.SILENCE: QColor(158, 158, 158),
    SegmentType.RECAP: QColor(255, 152, 0),
    SegmentType.UNKNOWN: QColor(96, 125, 139)
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


class VideoThread(QThread):
    """Thread for video frame capture."""
    frame_ready = pyqtSignal(np.ndarray, float)
    finished_playing = pyqtSignal()
    
    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self.cap = None
        self.playing = False
        self.seek_to = -1
        self.fps = 30
        self.duration = 0
        self.current_pos = 0
        self._stop = False
    
    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = frame_count / self.fps
        
        frame_delay = int(1000 / self.fps)
        
        while not self._stop:
            if self.seek_to >= 0:
                frame_num = int(self.seek_to * self.fps)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                self.seek_to = -1
            
            if self.playing:
                ret, frame = self.cap.read()
                if ret:
                    self.current_pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame_ready.emit(frame_rgb, self.current_pos)
                else:
                    self.playing = False
                    self.finished_playing.emit()
            
            self.msleep(frame_delay)
        
        self.cap.release()
    
    def play(self):
        self.playing = True
    
    def pause(self):
        self.playing = False
    
    def seek(self, position: float):
        self.seek_to = position
    
    def stop(self):
        self._stop = True


class VideoDisplay(QLabel):
    """Widget to display video frames."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
    
    def display_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        scaled = qimg.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(scaled))


class SegmentTimeline(QWidget):
    """Interactive timeline widget."""
    
    seekRequested = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments: List[VideoSegment] = []
        self.duration = 0.0
        self.current_position = 0.0
        
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
    
    def set_segments(self, segments: List[VideoSegment], duration: float):
        self.segments = segments
        self.duration = duration
        self.update()
    
    def set_position(self, position: float):
        self.current_position = position
        self.update()
    
    def paintEvent(self, event):
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
            triangle = [
                QPoint(pos_x, y_offset - 3),
                QPoint(pos_x - 6, y_offset - 9),
                QPoint(pos_x + 6, y_offset - 9)
            ]
            painter.drawPolygon(*triangle)
        
        painter.setPen(QPen(Qt.white, 1))
        painter.setFont(QFont('Arial', 8))
        
        for i in range(11):
            x = x_offset + int((i / 10) * width)
            time = (i / 10) * self.duration
            painter.drawLine(x, y_offset + height, x, y_offset + height + 3)
            
            time_str = f"{int(time // 60):02d}:{int(time % 60):02d}"
            fm = QFontMetrics(painter.font())
            painter.drawText(x - fm.horizontalAdvance(time_str) // 2, y_offset + height + 15, time_str)
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            position = self._get_time_at_x(event.x())
            if 0 <= position <= self.duration:
                self.seekRequested.emit(position)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        position = self._get_time_at_x(event.x())
        if 0 <= position <= self.duration:
            segment = self._get_segment_at_time(position)
            if segment:
                tooltip = (
                    f"{SEGMENT_LABELS[segment.segment_type]}\n"
                    f"Time: {int(position // 60):02d}:{int(position % 60):02d}\n"
                    f"Duration: {segment.duration:.1f}s"
                )
                QToolTip.showText(event.globalPos(), tooltip)
    
    def _get_time_at_x(self, x: int) -> float:
        width = self.width() - 20
        ratio = (x - 10) / width
        return ratio * self.duration
    
    def _get_segment_at_time(self, time: float) -> Optional[VideoSegment]:
        for segment in self.segments:
            if segment.start_time <= time < segment.end_time:
                return segment
        return None


class SegmentListWidget(QWidget):
    """Segment list panel."""
    
    seekRequested = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments: List[VideoSegment] = []
        self.current_idx = -1
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        self.buttons: List[QPushButton] = []
    
    def set_segments(self, segments: List[VideoSegment]):
        for btn in self.buttons:
            btn.deleteLater()
        self.buttons.clear()
        self.segments = segments
        
        for i, seg in enumerate(segments):
            color = SEGMENT_COLORS.get(seg.segment_type, QColor(128, 128, 128))
            label = SEGMENT_LABELS[seg.segment_type]
            
            time_str = f"{int(seg.start_time // 60):02d}:{int(seg.start_time % 60):02d}"
            btn = QPushButton(f"{time_str} - {label} ({seg.duration:.1f}s)")
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
            self.buttons.append(btn)
        
        self.layout().addStretch()
    
    def highlight_segment(self, time: float):
        for i, seg in enumerate(self.segments):
            if seg.start_time <= time < seg.end_time:
                if i != self.current_idx:
                    self.current_idx = i
                    self._update_highlights()
                break
    
    def _update_highlights(self):
        for i, btn in enumerate(self.buttons):
            if i < len(self.segments):
                seg = self.segments[i]
                color = SEGMENT_COLORS.get(seg.segment_type, QColor(128, 128, 128))
                
                border = "3px solid white" if i == self.current_idx else "none"
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {color.name()};
                        color: white;
                        border: {border};
                        padding: 5px;
                        text-align: left;
                        border-radius: 3px;
                    }}
                    QPushButton:hover {{
                        background-color: {color.lighter(120).name()};
                    }}
                """)


class OpenCVVideoPlayer(QMainWindow):
    """Main video player using OpenCV backend."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("CSCI 576 - Video Segmentation Player")
        self.setGeometry(100, 100, 1280, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: white; }
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
            QCheckBox { color: white; }
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
        """)
        
        self.video_thread: Optional[VideoThread] = None
        self.segmentation_result: Optional[SegmentationResult] = None
        self.auto_skip_enabled = False
        self.current_video_path = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        self.video_display = VideoDisplay()
        left_layout.addWidget(self.video_display)
        
        self.timeline = SegmentTimeline()
        self.timeline.seekRequested.connect(self._seek)
        left_layout.addWidget(self.timeline)
        
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("Content (green)"))
        legend_layout.addWidget(QLabel("Advertisement (red)"))
        legend_layout.addStretch()
        left_layout.addLayout(legend_layout)
        
        controls = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop)
        controls.addWidget(self.stop_btn)
        
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self._prev_segment)
        controls.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._next_segment)
        controls.addWidget(self.next_btn)
        
        self.skip_btn = QPushButton("Skip Ad ⏭")
        self.skip_btn.setStyleSheet("""
            QPushButton { background-color: #f44336; font-weight: bold; }
            QPushButton:hover { background-color: #e53935; }
        """)
        self.skip_btn.clicked.connect(self._skip_ad)
        controls.addWidget(self.skip_btn)
        
        controls.addSpacing(20)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(120)
        controls.addWidget(self.time_label)
        
        controls.addStretch()
        
        self.auto_skip_cb = QCheckBox("Auto-skip ads")
        self.auto_skip_cb.stateChanged.connect(lambda s: setattr(self, 'auto_skip_enabled', s == Qt.Checked))
        controls.addWidget(self.auto_skip_cb)
        
        left_layout.addLayout(controls)
        main_layout.addWidget(left_panel, stretch=3)
        
        right_panel = QWidget()
        right_panel.setMaximumWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        load_group = QGroupBox("Load Video")
        load_layout = QVBoxLayout(load_group)
        
        self.load_btn = QPushButton("Open Video File")
        self.load_btn.clicked.connect(self._open_file)
        load_layout.addWidget(self.load_btn)
        
        right_layout.addWidget(load_group)
        
        stats_group = QGroupBox("Video Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("No video loaded")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        right_layout.addWidget(stats_group)
        
        segments_group = QGroupBox("Segments")
        segments_layout = QVBoxLayout(segments_group)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        self.segment_list = SegmentListWidget()
        self.segment_list.seekRequested.connect(self._seek)
        scroll.setWidget(self.segment_list)
        
        segments_layout.addWidget(scroll)
        right_layout.addWidget(segments_group, stretch=1)
        
        main_layout.addWidget(right_panel, stretch=1)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready - Click 'Open Video File' to start")
    
    def load_video(self, video_path: str):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "Error", f"File not found: {video_path}")
            return
        
        self.current_video_path = video_path
        self.video_thread = VideoThread(video_path)
        self.video_thread.frame_ready.connect(self._on_frame)
        self.video_thread.finished_playing.connect(self._on_finished)
        self.video_thread.start()
        
        QTimer.singleShot(200, self._init_video_info)
        
        self.setWindowTitle(f"CSCI 576 Player - {os.path.basename(video_path)}")
        self.statusBar.showMessage(f"Loaded: {os.path.basename(video_path)}")
        
        info_path = self._find_info_file(video_path)
        if info_path:
            self._load_ground_truth(video_path, info_path)
    
    def _init_video_info(self):
        if self.video_thread:
            self.timeline.duration = self.video_thread.duration
            self.timeline.update()
    
    def _on_frame(self, frame: np.ndarray, position: float):
        self.video_display.display_frame(frame)
        self.timeline.set_position(position)
        
        duration = self.video_thread.duration if self.video_thread else 0
        self.time_label.setText(
            f"{int(position // 60):02d}:{int(position % 60):02d} / "
            f"{int(duration // 60):02d}:{int(duration % 60):02d}"
        )
        
        self.segment_list.highlight_segment(position)
        
        if self.auto_skip_enabled and self.segmentation_result:
            for seg in self.segmentation_result.segments:
                if seg.start_time <= position < seg.start_time + 0.5:
                    if seg.segment_type == SegmentType.AD:
                        self._seek(seg.end_time)
                        self.statusBar.showMessage(f"Auto-skipped advertisement")
                        break
    
    def _on_finished(self):
        self.play_btn.setText("▶ Play")
    
    def _toggle_play(self):
        if not self.video_thread:
            return
        
        if self.video_thread.playing:
            self.video_thread.pause()
            self.play_btn.setText("▶ Play")
        else:
            self.video_thread.play()
            self.play_btn.setText("⏸ Pause")
    
    def _stop(self):
        if self.video_thread:
            self.video_thread.pause()
            self.video_thread.seek(0)
            self.play_btn.setText("▶ Play")
    
    def _seek(self, position: float):
        if self.video_thread:
            self.video_thread.seek(position)
    
    def _prev_segment(self):
        if not self.segmentation_result or not self.video_thread:
            return
        
        current = self.video_thread.current_pos
        for seg in reversed(self.segmentation_result.segments):
            if seg.end_time < current - 0.5:
                self._seek(seg.start_time)
                return
        self._seek(0)
    
    def _next_segment(self):
        if not self.segmentation_result or not self.video_thread:
            return
        
        current = self.video_thread.current_pos
        for seg in self.segmentation_result.segments:
            if seg.start_time > current + 0.5:
                self._seek(seg.start_time)
                return
    
    def _skip_ad(self):
        if not self.segmentation_result or not self.video_thread:
            return
        
        current = self.video_thread.current_pos
        for seg in self.segmentation_result.segments:
            if seg.start_time <= current < seg.end_time:
                if seg.segment_type in [SegmentType.AD, SegmentType.INTRO, SegmentType.OUTRO]:
                    self._seek(seg.end_time)
                    self.statusBar.showMessage(f"Skipped {SEGMENT_LABELS[seg.segment_type]}")
                return
    
    def _open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if file_path:
            self.load_video(file_path)
    
    def _find_info_file(self, video_path: str) -> Optional[str]:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        paths = [
            os.path.join(video_dir, '..', 'video_info', f'{video_name}.json'),
            os.path.join(video_dir, 'video_info', f'{video_name}.json'),
        ]
        
        for path in paths:
            normalized = os.path.normpath(path)
            if os.path.exists(normalized):
                return normalized
        return None
    
    def _load_ground_truth(self, video_path: str, info_path: str):
        try:
            try:
                from .segmentation import GroundTruthSegmenter
            except ImportError:
                from segmentation import GroundTruthSegmenter
            
            segmenter = GroundTruthSegmenter(video_path, info_path)
            result = segmenter.get_segmentation()
            self.set_segmentation(result)
            self.statusBar.showMessage(f"Loaded with ground truth segmentation")
        except Exception as e:
            self.statusBar.showMessage(f"Could not load ground truth: {e}")
    
    def set_segmentation(self, result: SegmentationResult):
        self.segmentation_result = result
        
        self.timeline.set_segments(result.segments, result.duration)
        self.segment_list.set_segments(result.segments)
        
        summary = result.to_dict()['summary']
        content_time = summary['type_durations'].get('core_content', 0)
        ad_time = summary['type_durations'].get('ad', 0)
        
        self.stats_label.setText(
            f"Duration: {int(result.duration // 60):02d}:{int(result.duration % 60):02d}\n"
            f"Segments: {len(result.segments)}\n"
            f"Content: {summary['content_ratio']:.1%} ({content_time:.0f}s)\n"
            f"Ads: {summary['ad_ratio']:.1%} ({ad_time:.0f}s)\n"
            f"Ads inserted: {result.metadata.get('num_ads', 0)}"
        )
    
    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        super().closeEvent(event)


def run_opencv_player(video_path: Optional[str] = None):
    """Launch the OpenCV-based video player."""
    app = QApplication(sys.argv)
    
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    app.setPalette(palette)
    
    player = OpenCVVideoPlayer()
    player.show()
    
    if video_path:
        player.load_video(video_path)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    run_opencv_player(video)
