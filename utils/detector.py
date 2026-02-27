"""
YOLOv5检测器封装
提供逐帧目标检测接口
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path


class YOLOv5Detector:
    """YOLOv5检测器封装类"""
    
    # COCO数据集类别ID映射
    COCO_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
    }
    
    # 目标类别名称到ID的映射
    CLASS_NAME_TO_ID = {
        'person': 0,
        'car': 2,
    }
    
    def __init__(self, model_path: str = None, model_size: str = 'n', 
                 conf_thres: float = 0.25, iou_thres: float = 0.45,
                 device: str = None):
        """
        初始化YOLOv5检测器
        
        Args:
            model_path: 模型路径，如果为None则自动加载
            model_size: 模型尺寸 ('n' for nano, 's' for small)
            conf_thres: 置信度阈值
            iou_thres: NMS IOU阈值
            device: 运行设备 ('cuda', 'cpu' 或 None自动选择)
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 加载模型
        self.model, self.yolo_wrapper = self._load_model(model_path, model_size)
        
        # 获取类别名称
        if self.yolo_wrapper:
            self.names = self.model.names
        else:
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
    def _load_model(self, model_path: str, model_size: str):
        """
        加载YOLOv5模型
        
        Args:
            model_path: 模型路径
            model_size: 模型尺寸
            
        Returns:
            (model, is_yolo_wrapper) 元组
        """
        if model_path is None:
            # 默认模型路径
            project_root = Path(__file__).parent.parent
            model_name = f'yolov5{model_size}u.pt'
            model_path = project_root / 'data' / 'weights' / model_name
            
        if isinstance(model_path, Path):
            model_path = str(model_path)
            
        # 优先使用ultralytics YOLO类加载 (自动处理预处理)
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            return model, True
        except Exception as e:
            print(f"ultralytics YOLO 加载失败: {e}, 尝试torch.hub...")
        
        # 回退到torch.hub加载
        try:
            model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}', 
                                   pretrained=True, device=self.device)
            return model, False
        except Exception as e:
            raise RuntimeError(f"无法加载YOLOv5模型: {e}")
    
    def detect(self, frame: np.ndarray, target_classes: list = None) -> list:
        """
        对单帧图像进行目标检测
        
        Args:
            frame: BGR格式的图像帧 (numpy数组)
            target_classes: 目标类别列表，如 ['person', 'car']，为None则检测所有类别
            
        Returns:
            检测结果列表，每个元素为 [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # 获取目标类别ID
        target_class_ids = None
        if target_classes:
            target_class_ids = [self.CLASS_NAME_TO_ID.get(cls, -1) for cls in target_classes]
            target_class_ids = [cid for cid in target_class_ids if cid >= 0]
        
        # 推理
        with torch.no_grad():
            if self.yolo_wrapper:
                # 使用ultralytics YOLO (自动处理预处理)
                results = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres, 
                                             device=self.device, verbose=False)
            else:
                # torch.hub加载的模型
                results = self.model(frame)
            
        # 解析结果
        detections = []
        
        # 处理不同格式的输出
        if self.yolo_wrapper:
            # ultralytics YOLO格式
            if len(results) > 0 and hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        conf = confs[i]
                        cls_id = cls_ids[i]
                        
                        # 过滤目标类别
                        if target_class_ids is not None and cls_id not in target_class_ids:
                            continue
                            
                        # 过滤置信度
                        if conf < self.conf_thres:
                            continue
                            
                        # 获取类别名称
                        cls_name = self.names[cls_id] if cls_id < len(self.names) else f'class_{cls_id}'
                        
                        detections.append([
                            float(x1), float(y1), float(x2), float(y2),
                            float(conf), cls_id, cls_name
                        ])
        elif hasattr(results, 'xyxy'):
            # torch.hub加载的模型
            pred = results.xyxy[0].cpu().numpy()
            for det in pred:
                if len(det) < 6:
                    continue
                    
                x1, y1, x2, y2, conf, cls_id = det[:6]
                cls_id = int(cls_id)
                
                # 过滤目标类别
                if target_class_ids is not None and cls_id not in target_class_ids:
                    continue
                    
                # 过滤置信度
                if conf < self.conf_thres:
                    continue
                    
                # 获取类别名称
                cls_name = self.names[cls_id] if cls_id < len(self.names) else f'class_{cls_id}'
                
                detections.append([
                    float(x1), float(y1), float(x2), float(y2),
                    float(conf), cls_id, cls_name
                ])
        
        return detections
    
    def get_bbox_tlwh(self, detections: list) -> np.ndarray:
        """
        将检测结果转换为tlwh格式 (top-left x, top-left y, width, height)
        
        Args:
            detections: detect()返回的检测结果
            
        Returns:
            tlwh格式的边界框数组
        """
        if not detections:
            return np.array([])
            
        bboxes = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            w, h = x2 - x1, y2 - y1
            bboxes.append([x1, y1, w, h])
            
        return np.array(bboxes)
    
    def get_confidences(self, detections: list) -> np.ndarray:
        """
        获取检测结果的置信度
        
        Args:
            detections: detect()返回的检测结果
            
        Returns:
            置信度数组
        """
        if not detections:
            return np.array([])
            
        return np.array([det[4] for det in detections])
    
    def get_class_ids(self, detections: list) -> np.ndarray:
        """
        获取检测结果的类别ID
        
        Args:
            detections: detect()返回的检测结果
            
        Returns:
            类别ID数组
        """
        if not detections:
            return np.array([])
            
        return np.array([int(det[5]) for det in detections])
    
    def get_class_names(self, detections: list) -> list:
        """
        获取检测结果的类别名称
        
        Args:
            detections: detect()返回的检测结果
            
        Returns:
            类别名称列表
        """
        if not detections:
            return []
            
        return [det[6] for det in detections]
