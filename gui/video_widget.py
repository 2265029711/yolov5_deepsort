"""
视频显示组件
用于显示检测结果视频帧
"""
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
import cv2
import numpy as np


class VideoWidget(QWidget):
    """视频显示控件"""
    
    def __init__(self, title: str = "视频", parent=None):
        super().__init__(parent)
        self.title = title
        self.current_frame = None
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # 标题标签
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.title_label.setStyleSheet("""
            QLabel {
                color: #1976D2;
                padding: 6px;
                background-color: #E3F2FD;
                border-radius: 4px;
            }
        """)
        
        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                color: #888;
                font-size: 16px;
            }
        """)
        self.video_label.setText("等待视频输入...")
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.video_label, 1)
        
    def update_frame(self, frame: np.ndarray):
        """
        更新显示的视频帧
        
        Args:
            frame: OpenCV格式的图像帧 (BGR)
        """
        if frame is None:
            return
            
        self.current_frame = frame.copy()
        
        # BGR转RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 获取帧尺寸
        h, w, ch = rgb_frame.shape
        
        # 转换为QImage
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放到合适大小
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        
    def clear(self):
        """清除显示"""
        self.current_frame = None
        self.video_label.clear()
        self.video_label.setText("等待视频输入...")
        
    def resizeEvent(self, event):
        """窗口大小改变时重新绘制当前帧"""
        if self.current_frame is not None:
            self.update_frame(self.current_frame)
        super().resizeEvent(event)