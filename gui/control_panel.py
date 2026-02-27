"""
控制面板组件
包含视频控制、跟踪选项、模型参数等功能
"""
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, 
    QCheckBox, QLabel, QComboBox, QGroupBox, QSlider,
    QFileDialog, QFrame, QScrollArea
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont


class ControlPanel(QWidget):
    """控制面板 - 左侧控制区域"""
    
    # 信号定义
    open_video_clicked = pyqtSignal()
    predict_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    track_person_changed = pyqtSignal(bool)
    track_car_changed = pyqtSignal(bool)
    conf_changed = pyqtSignal(float)
    progress_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path = ""
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        
        # ========== 视频源设置 ==========
        video_group = QGroupBox("视频源")
        video_group.setStyleSheet(self._get_group_style())
        video_layout = QVBoxLayout(video_group)
        video_layout.setSpacing(8)
        
        path_label = QLabel("视频路径：")
        path_label.setStyleSheet("color: #555; font-size: 12px;")
        
        self.path_display = QLabel("未选择视频")
        self.path_display.setWordWrap(True)
        self.path_display.setStyleSheet("""
            QLabel {
                background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 8px;
                color: #333;
                font-size: 11px;
            }
        """)
        
        self.btn_open = QPushButton("选择视频")
        self.btn_open.setFixedHeight(32)
        self.btn_open.setStyleSheet(self._get_button_style())
        
        video_layout.addWidget(path_label)
        video_layout.addWidget(self.path_display)
        video_layout.addWidget(self.btn_open)
        
        # ========== 模型参数设置 ==========
        model_group = QGroupBox("模型参数")
        model_group.setStyleSheet(self._get_group_style())
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(10)
        
        model_label = QLabel("检测模型：")
        model_label.setStyleSheet("color: #555; font-size: 12px;")
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv5n (最快)", "YOLOv5s", "YOLOv5m", "YOLOv5l"])
        self.model_combo.setStyleSheet(self._get_combo_style())
        
        conf_label = QLabel("置信度阈值：")
        conf_label.setStyleSheet("color: #555; font-size: 12px;")
        
        conf_layout = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(25)
        self.conf_slider.setStyleSheet(self._get_slider_style())
        
        self.conf_value = QLabel("0.25")
        self.conf_value.setFixedWidth(40)
        self.conf_value.setStyleSheet("color: #1976D2; font-weight: bold;")
        
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value)
        
        self.model_info = QLabel("模型加载后显示mAP等指标")
        self.model_info.setStyleSheet("""
            QLabel {
                background-color: #E3F2FD;
                border: 1px solid #90CAF9;
                border-radius: 4px;
                padding: 6px;
                color: #1565C0;
                font-size: 11px;
            }
        """)
        self.model_info.setWordWrap(True)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(conf_label)
        model_layout.addLayout(conf_layout)
        model_layout.addWidget(self.model_info)
        
        # ========== 跟踪对象选择 ==========
        track_group = QGroupBox("跟踪对象")
        track_group.setStyleSheet(self._get_group_style())
        track_layout = QVBoxLayout(track_group)
        track_layout.setSpacing(8)
        
        self.chk_person = QCheckBox("行人 (person)")
        self.chk_person.setChecked(True)
        self.chk_person.setStyleSheet(self._get_checkbox_style())
        
        self.chk_car = QCheckBox("车辆 (car)")
        self.chk_car.setChecked(False)
        self.chk_car.setStyleSheet(self._get_checkbox_style())
        
        track_layout.addWidget(self.chk_person)
        track_layout.addWidget(self.chk_car)
        
        # ========== 控制按钮 ==========
        control_group = QGroupBox("控制")
        control_group.setStyleSheet(self._get_group_style())
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(10)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.btn_predict = QPushButton("开始")
        self.btn_predict.setFixedSize(70, 36)
        self.btn_predict.setStyleSheet(self._get_primary_button_style())
        self.btn_predict.setEnabled(False)
        
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setFixedSize(70, 36)
        self.btn_pause.setStyleSheet(self._get_warning_button_style())
        self.btn_pause.setEnabled(False)
        
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFixedSize(70, 36)
        self.btn_stop.setStyleSheet(self._get_danger_button_style())
        self.btn_stop.setEnabled(False)
        
        btn_layout.addWidget(self.btn_predict)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()
        
        control_layout.addLayout(btn_layout)
        
        # ========== 实时参数显示 ==========
        realtime_group = QGroupBox("实时参数")
        realtime_group.setStyleSheet(self._get_group_style())
        realtime_layout = QVBoxLayout(realtime_group)
        realtime_layout.setSpacing(6)
        
        # FPS
        fps_layout = QHBoxLayout()
        fps_label = QLabel("处理速度:")
        fps_label.setStyleSheet("color: #555; font-size: 12px;")
        self.fps_value = QLabel("0 FPS")
        self.fps_value.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 13px;")
        fps_layout.addWidget(fps_label)
        fps_layout.addStretch()
        fps_layout.addWidget(self.fps_value)
        
        # 检测数量
        det_layout = QHBoxLayout()
        det_label = QLabel("检测数量:")
        det_label.setStyleSheet("color: #555; font-size: 12px;")
        self.det_value = QLabel("0")
        self.det_value.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 13px;")
        det_layout.addWidget(det_label)
        det_layout.addStretch()
        det_layout.addWidget(self.det_value)
        
        # 跟踪数量
        track_layout = QHBoxLayout()
        track_label = QLabel("跟踪目标:")
        track_label.setStyleSheet("color: #555; font-size: 12px;")
        self.track_value = QLabel("0")
        self.track_value.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 13px;")
        track_layout.addWidget(track_label)
        track_layout.addStretch()
        track_layout.addWidget(self.track_value)
        
        # 行人数量
        person_layout = QHBoxLayout()
        person_label = QLabel("行人:")
        person_label.setStyleSheet("color: #555; font-size: 12px;")
        self.person_value = QLabel("0")
        self.person_value.setStyleSheet("color: #9C27B0; font-weight: bold; font-size: 13px;")
        person_layout.addWidget(person_label)
        person_layout.addStretch()
        person_layout.addWidget(self.person_value)
        
        # 车辆数量
        car_layout = QHBoxLayout()
        car_label = QLabel("车辆:")
        car_label.setStyleSheet("color: #555; font-size: 12px;")
        self.car_value = QLabel("0")
        self.car_value.setStyleSheet("color: #00BCD4; font-weight: bold; font-size: 13px;")
        car_layout.addWidget(car_label)
        car_layout.addStretch()
        car_layout.addWidget(self.car_value)
        
        # 视频分辨率
        res_layout = QHBoxLayout()
        res_label = QLabel("分辨率:")
        res_label.setStyleSheet("color: #555; font-size: 12px;")
        self.res_value = QLabel("--")
        self.res_value.setStyleSheet("color: #607D8B; font-size: 12px;")
        res_layout.addWidget(res_label)
        res_layout.addStretch()
        res_layout.addWidget(self.res_value)
        
        realtime_layout.addLayout(fps_layout)
        realtime_layout.addLayout(det_layout)
        realtime_layout.addLayout(track_layout)
        realtime_layout.addLayout(person_layout)
        realtime_layout.addLayout(car_layout)
        realtime_layout.addLayout(res_layout)
        
        # ========== 添加到主布局 ==========
        main_layout.addWidget(video_group)
        main_layout.addWidget(model_group)
        main_layout.addWidget(track_group)
        main_layout.addWidget(control_group)
        main_layout.addWidget(realtime_group)
        main_layout.addStretch()
        
        self._connect_signals()
        
    def _connect_signals(self):
        self.btn_open.clicked.connect(self._on_open_video)
        self.btn_predict.clicked.connect(self.predict_clicked)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self.stop_clicked)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        self.chk_person.stateChanged.connect(
            lambda state: self.track_person_changed.emit(state == Qt.Checked)
        )
        self.chk_car.stateChanged.connect(
            lambda state: self.track_car_changed.emit(state == Qt.Checked)
        )
        
    def _on_open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        
        if file_path:
            self.video_path = file_path
            import os
            filename = os.path.basename(file_path)
            self.path_display.setText(filename)
            self.path_display.setToolTip(file_path)
            self.open_video_clicked.emit()
            
    def _on_pause(self):
        if self.btn_pause.text() == "暂停":
            self.btn_pause.setText("继续")
        else:
            self.btn_pause.setText("暂停")
        self.pause_clicked.emit()
        
    def _on_conf_changed(self, value):
        conf = value / 100.0
        self.conf_value.setText(f"{conf:.2f}")
        self.conf_changed.emit(conf)
        
    def set_predict_enabled(self, enabled: bool):
        self.btn_predict.setEnabled(enabled)
        
    def set_pause_enabled(self, enabled: bool):
        self.btn_pause.setEnabled(enabled)
        if not enabled:
            self.btn_pause.setText("暂停")
        
    def set_stop_enabled(self, enabled: bool):
        self.btn_stop.setEnabled(enabled)
        
    def set_model_info(self, info: str):
        self.model_info.setText(info)
        
    def update_realtime_stats(self, stats: dict):
        if 'fps' in stats:
            self.fps_value.setText(f"{stats['fps']:.1f} FPS")
        if 'detections' in stats:
            self.det_value.setText(str(stats['detections']))
        if 'total' in stats:
            self.track_value.setText(str(stats['total']))
        if 'person' in stats:
            self.person_value.setText(str(stats['person']))
        if 'car' in stats:
            self.car_value.setText(str(stats['car']))
            
    def set_resolution(self, width: int, height: int):
        self.res_value.setText(f"{width}x{height}")
        
    def reset_stats(self):
        self.fps_value.setText("0 FPS")
        self.det_value.setText("0")
        self.track_value.setText("0")
        self.person_value.setText("0")
        self.car_value.setText("0")
        
    def get_track_options(self) -> dict:
        return {
            'track_person': self.chk_person.isChecked(),
            'track_car': self.chk_car.isChecked(),
        }
        
    def get_conf_threshold(self) -> float:
        return self.conf_slider.value() / 100.0
        
    def get_model_size(self) -> str:
        model_map = {0: 'n', 1: 's', 2: 'm', 3: 'l'}
        return model_map.get(self.model_combo.currentIndex(), 'n')
        
    def _get_group_style(self) -> str:
        return """
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #1976D2;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """
        
    def _get_button_style(self) -> str:
        return """
            QPushButton {
                background-color: #E0E0E0;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                color: #424242;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #E3F2FD;
                border-color: #90CAF9;
            }
            QPushButton:pressed {
                background-color: #BBDEFB;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #BDBDBD;
                border-color: #E0E0E0;
            }
        """
        
    def _get_primary_button_style(self) -> str:
        return """
            QPushButton {
                background-color: #1976D2;
                border: 1px solid #1565C0;
                border-radius: 4px;
                color: white;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1E88E5;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #E0E0E0;
                border-color: #BDBDBD;
            }
        """
        
    def _get_warning_button_style(self) -> str:
        return """
            QPushButton {
                background-color: #FF9800;
                border: 1px solid #F57C00;
                border-radius: 4px;
                color: white;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #FFA726;
            }
            QPushButton:pressed {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #E0E0E0;
                border-color: #BDBDBD;
            }
        """
        
    def _get_danger_button_style(self) -> str:
        return """
            QPushButton {
                background-color: #E53935;
                border: 1px solid #C62828;
                border-radius: 4px;
                color: white;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #EF5350;
            }
            QPushButton:pressed {
                background-color: #C62828;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #E0E0E0;
                border-color: #BDBDBD;
            }
        """
        
    def _get_checkbox_style(self) -> str:
        return """
            QCheckBox {
                font-size: 12px;
                color: #333333;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #BDBDBD;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #1976D2;
                border-radius: 3px;
                background-color: #1976D2;
            }
            QCheckBox::indicator:unchecked:hover {
                border-color: #1976D2;
            }
        """
        
    def _get_combo_style(self) -> str:
        return """
            QComboBox {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 6px 12px;
                background-color: white;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #90CAF9;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #666;
            }
        """
        
    def _get_slider_style(self) -> str:
        return """
            QSlider::groove:horizontal {
                border: 1px solid #E0E0E0;
                height: 6px;
                background: #F5F5F5;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #1976D2;
                border: 1px solid #1565C0;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #1E88E5;
            }
            QSlider::sub-page:horizontal {
                background: #90CAF9;
                border-radius: 3px;
            }
        """


class VideoProgressBar(QWidget):
    """视频进度条组件"""
    
    position_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30.0
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        
        self.time_label = QLabel("00:00")
        self.time_label.setFixedWidth(50)
        self.time_label.setStyleSheet("color: #666; font-size: 12px;")
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.setValue(0)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #E0E0E0;
                height: 8px;
                background: #F5F5F5;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #1976D2;
                border: 2px solid #1565C0;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1E88E5;
            }
            QSlider::sub-page:horizontal {
                background: #1976D2;
                border-radius: 4px;
            }
        """)
        self.progress_slider.setEnabled(False)
        
        self.total_label = QLabel("00:00")
        self.total_label.setFixedWidth(50)
        self.total_label.setStyleSheet("color: #666; font-size: 12px;")
        
        layout.addWidget(self.time_label)
        layout.addWidget(self.progress_slider, 1)
        layout.addWidget(self.total_label)
        
        self.progress_slider.valueChanged.connect(self._on_slider_changed)
        
    def _on_slider_changed(self, value):
        if self.total_frames > 0:
            frame = int(value * self.total_frames / 100)
            self.current_frame = frame
            self._update_time_label()
            self.position_changed.emit(frame)
            
    def set_total_frames(self, total: int, fps: float = 30.0):
        self.total_frames = total
        self.fps = fps if fps > 0 else 30.0
        self._update_total_label()
        
    def set_current_frame(self, frame: int):
        self.current_frame = frame
        if self.total_frames > 0:
            self.progress_slider.blockSignals(True)
            self.progress_slider.setValue(int(frame * 100 / self.total_frames))
            self.progress_slider.blockSignals(False)
        self._update_time_label()
        
    def set_enabled(self, enabled: bool):
        self.progress_slider.setEnabled(enabled)
        
    def _update_time_label(self):
        if self.fps > 0:
            seconds = self.current_frame / self.fps
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            self.time_label.setText(f"{mins:02d}:{secs:02d}")
            
    def _update_total_label(self):
        if self.fps > 0 and self.total_frames > 0:
            seconds = self.total_frames / self.fps
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            self.total_label.setText(f"{mins:02d}:{secs:02d}")
        
    def reset(self):
        self.current_frame = 0
        self.progress_slider.setValue(0)
        self.time_label.setText("00:00")