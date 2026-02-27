"""
主窗口界面
YOLOv5+DeepSORT多目标跟踪可视化界面
布局：左侧控制面板 + 右侧视频显示 + 底部进度条
"""
import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QMessageBox, QStatusBar, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from .video_widget import VideoWidget
from .control_panel import ControlPanel, VideoProgressBar

# 导入检测器和跟踪器
from utils.detector import YOLOv5Detector
from utils.tracker import DeepSORTTracker


class VideoThread(QThread):
    """视频处理线程"""
    
    frame_ready = pyqtSignal(np.ndarray)  # 处理后帧
    stats_updated = pyqtSignal(dict)  # 统计信息更新
    frame_position = pyqtSignal(int)  # 当前帧位置
    finished = pyqtSignal()
    
    def __init__(self, video_path: str, track_options: dict = None, 
                 model_size: str = 'n', conf_thres: float = 0.25, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.track_options = track_options or {}
        self.model_size = model_size
        self.conf_thres = conf_thres
        self.running = False
        self.paused = False
        
        # 初始化检测器和跟踪器
        self.detector = None
        self.tracker = None
        
        # FPS计算
        self.frame_times = []
        
    def init_models(self):
        """初始化检测和跟踪模型"""
        try:
            # 初始化YOLOv5检测器
            self.detector = YOLOv5Detector(
                model_size=self.model_size,
                conf_thres=self.conf_thres,
                iou_thres=0.45
            )
            
            # 初始化DeepSORT跟踪器
            self.tracker = DeepSORTTracker(
                max_cosine_distance=0.2,
                max_age=30,
                n_init=3
            )
            
            return True
        except Exception as e:
            print(f"模型初始化失败: {e}")
            return False
        
    def run(self):
        """运行视频处理"""
        # 初始化模型
        if not self.init_models():
            self.finished.emit()
            return
            
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            self.finished.emit()
            return
            
        self.running = True
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30
        
        # 获取目标类别
        target_classes = []
        if self.track_options.get('track_person', False):
            target_classes.append('person')
        if self.track_options.get('track_car', False):
            target_classes.append('car')
            
        # 颜色映射 (每个track_id一个颜色)
        color_map = {}
        frame_count = 0
        
        while self.running:
            if self.paused:
                self.msleep(100)
                continue
                
            # 计算FPS
            start_time = time.time()
                
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 目标检测
            detections = self.detector.detect(frame, target_classes if target_classes else None)
            
            # 目标跟踪
            tracks = self.tracker.update(frame, detections)
            
            # 绘制结果
            processed_frame = self._draw_results(frame.copy(), tracks, color_map)
            
            # 计算处理时间
            process_time = time.time() - start_time
            self.frame_times.append(process_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            
            # 计算平均FPS
            avg_time = sum(self.frame_times) / len(self.frame_times)
            current_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # 发送帧和统计信息
            self.frame_ready.emit(processed_frame)
            self.frame_position.emit(frame_count)
            
            # 发送统计信息
            stats = self.tracker.get_track_count()
            stats['total'] = len(tracks)
            stats['frame'] = frame_count
            stats['total_frames'] = self.total_frames
            stats['fps'] = current_fps
            stats['detections'] = len(detections)
            self.stats_updated.emit(stats)
            
            self.msleep(delay)
            
        cap.release()
        self.finished.emit()
        
    def _draw_results(self, frame: np.ndarray, tracks: list, color_map: dict) -> np.ndarray:
        """
        在帧上绘制检测结果
        """
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            class_name = track['class_name']
            confidence = track['confidence']
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # 为每个track_id分配颜色
            if track_id not in color_map:
                hue = (track_id * 41) % 180
                color = [int(c) for c in cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
                )[0][0]]
                color_map[track_id] = color
            else:
                color = color_map[track_id]
                
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label = f"#{track_id} {class_name}"
            if confidence > 0:
                label += f" {confidence:.2f}"
                
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1
            )
            
            # 绘制标签文字
            cv2.putText(
                frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
        return frame
        
    def stop(self):
        """停止视频处理"""
        self.running = False
        self.wait()
        
    def pause(self):
        """暂停/继续"""
        self.paused = not self.paused
        return self.paused


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.video_thread = None
        self.current_video_path = None
        self.video_info = {}
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        # 设置窗口
        self.setWindowTitle("毕业设计 - YOLOv5+DeepSORT多目标跟踪系统")
        self.setMinimumSize(1280, 870)
        self.resize(1280, 870)
        
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FAFAFA;
            }
        """)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 标题栏
        title_bar = QWidget()
        title_bar.setFixedHeight(50)
        title_bar.setStyleSheet("""
            QWidget {
                background-color: #1976D2;
            }
        """)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 0, 20, 0)
        
        title_label = QLabel("毕业设计 - YOLOv5 + DeepSORT 多目标跟踪系统")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # 内容区域 (使用水平分割器)
        content_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        self.control_panel = ControlPanel()
        self.control_panel.setMinimumWidth(280)
        self.control_panel.setMaximumWidth(350)
        
        # 右侧视频显示区域
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(8, 8, 8, 8)
        video_layout.setSpacing(8)
        
        # 视频显示组件 (仅输出)
        self.output_video = VideoWidget("检测结果")
        
        # 统计信息显示
        stats_container = QWidget()
        stats_container.setFixedHeight(30)
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setContentsMargins(12, 0, 12, 0)
        
        self.stats_label = QLabel("就绪")
        self.stats_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 12px;
            }
        """)
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        
        video_layout.addWidget(self.output_video, 1)
        video_layout.addWidget(stats_container)
        
        # 进度条
        self.progress_bar = VideoProgressBar()
        self.progress_bar.setFixedHeight(40)
        
        # 添加到分割器
        content_splitter.addWidget(self.control_panel)
        content_splitter.addWidget(video_container)
        content_splitter.setSizes([300, 800])
        
        # 添加到主布局
        main_layout.addWidget(title_bar)
        main_layout.addWidget(content_splitter, 1)
        main_layout.addWidget(self.progress_bar)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请选择视频文件开始")
        
        # 连接信号
        self._connect_signals()
        
    def _connect_signals(self):
        """连接信号"""
        self.control_panel.open_video_clicked.connect(self._on_open_video)
        self.control_panel.predict_clicked.connect(self._on_predict)
        self.control_panel.pause_clicked.connect(self._on_pause)
        self.control_panel.stop_clicked.connect(self._on_stop)
        
    def _on_open_video(self):
        """打开视频文件"""
        file_path = self.control_panel.video_path
        
        if file_path:
            self.current_video_path = file_path
            
            # 获取视频信息
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                self.video_info = {
                    'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                }
                
                # 显示第一帧
                ret, frame = cap.read()
                if ret:
                    self.output_video.update_frame(frame)
                    
                # 更新进度条
                self.progress_bar.set_total_frames(
                    self.video_info['total_frames'],
                    self.video_info['fps']
                )
                
                cap.release()
                
            self.control_panel.set_predict_enabled(True)
            self.progress_bar.set_enabled(True)
            self.control_panel.set_resolution(
                self.video_info['width'], 
                self.video_info['height']
            )
            self.status_bar.showMessage(f"已加载: {file_path}")
            
    def _on_predict(self):
        """开始预测"""
        if not self.current_video_path:
            QMessageBox.warning(self, "警告", "请先选择视频文件")
            return
            
        track_options = self.control_panel.get_track_options()
        
        # 检查是否至少选择了一个跟踪对象
        if not any(track_options.values()):
            QMessageBox.warning(self, "警告", "请至少选择一个跟踪对象")
            return
            
        # 获取参数
        model_size = self.control_panel.get_model_size()
        conf_thres = self.control_panel.get_conf_threshold()
        
        # 更新模型信息
        self.control_panel.set_model_info(f"模型: YOLOv5{model_size}\n置信度: {conf_thres}")
        
        # 启动视频处理线程
        self.video_thread = VideoThread(
            self.current_video_path, 
            track_options,
            model_size,
            conf_thres
        )
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.stats_updated.connect(self._on_stats_updated)
        self.video_thread.frame_position.connect(self._on_frame_position)
        self.video_thread.finished.connect(self._on_video_finished)
        self.video_thread.start()
        
        # 更新按钮状态
        self.control_panel.set_predict_enabled(False)
        self.control_panel.set_pause_enabled(True)
        self.control_panel.set_stop_enabled(True)
        self.status_bar.showMessage("正在处理...")
        
    def _on_pause(self):
        """暂停/继续"""
        if self.video_thread:
            is_paused = self.video_thread.pause()
            if is_paused:
                self.status_bar.showMessage("已暂停")
            else:
                self.status_bar.showMessage("继续处理...")
                
    def _on_stop(self):
        """停止预测"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            
        self.control_panel.set_predict_enabled(True)
        self.control_panel.set_pause_enabled(False)
        self.control_panel.set_stop_enabled(False)
        self.control_panel.reset_stats()
        self.status_bar.showMessage("已停止")
        
    def _on_frame_ready(self, processed_frame: np.ndarray):
        """帧数据就绪"""
        self.output_video.update_frame(processed_frame)
        
    def _on_frame_position(self, frame_num: int):
        """帧位置更新"""
        self.progress_bar.set_current_frame(frame_num)
        
    def _on_video_finished(self):
        """视频处理完成"""
        self.control_panel.set_predict_enabled(True)
        self.control_panel.set_pause_enabled(False)
        self.control_panel.set_stop_enabled(False)
        self.control_panel.reset_stats()
        self.status_bar.showMessage("处理完成")
        
    def _on_stats_updated(self, stats: dict):
        """统计信息更新"""
        # 更新控制面板的实时参数
        self.control_panel.update_realtime_stats(stats)
        
        # 更新状态栏
        parts = [f"帧: {stats.get('frame', 0)}/{stats.get('total_frames', 0)}"]
        parts.append(f"跟踪总数: {stats.get('total', 0)}")
        
        if stats.get('person', 0) > 0:
            parts.append(f"行人: {stats['person']}")
        if stats.get('car', 0) > 0:
            parts.append(f"车辆: {stats['car']}")
            
        self.stats_label.setText(" | ".join(parts))
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.video_thread:
            self.video_thread.stop()
        event.accept()