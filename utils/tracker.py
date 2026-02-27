"""
DeepSORT跟踪器封装
提供多目标跟踪接口
"""
import os
import sys
import numpy as np
import cv2
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker


class DeepSORTTracker:
    """DeepSORT跟踪器封装类"""
    
    def __init__(self, model_path: str = None, max_cosine_distance: float = 0.2,
                 nn_budget: int = None, max_age: int = 30, n_init: int = 3):
        """
        初始化DeepSORT跟踪器
        
        Args:
            model_path: 特征提取模型路径 (mars-small128.pb)
            max_cosine_distance: 余弦距离阈值
            nn_budget: 特征库大小限制
            max_age: 跟踪目标最大丢失帧数
            n_init: 确认跟踪所需连续检测帧数
        """
        # 设置默认模型路径
        if model_path is None:
            model_path = PROJECT_ROOT / 'data' / 'weights' / 'mars-small128.pb'
            
        self.model_path = str(model_path)
        
        # 初始化特征提取器
        self.encoder = self._init_encoder()
        
        # 初始化距离度量
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        
        # 初始化跟踪器
        self.tracker = Tracker(metric, max_age=max_age, n_init=n_init)
        
        # 跟踪结果
        self.tracks = []
        
    def _init_encoder(self):
        """
        初始化特征提取器
        
        Returns:
            特征提取函数
        """
        try:
            # 尝试使用TensorFlow加载模型
            import tensorflow as tf
            
            # 使用TensorFlow 1.x兼容模式
            tf.compat.v1.disable_eager_execution()
            
            session = tf.compat.v1.Session()
            
            with tf.compat.v1.gfile.GFile(self.model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                
            tf.import_graph_def(graph_def, name="net")
            
            input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
                "images:0"
            )
            output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
                "features:0"
            )
            
            image_shape = input_var.get_shape().as_list()[1:]
            feature_dim = output_var.get_shape().as_list()[-1]
            
            def encoder(image, boxes):
                """特征提取函数"""
                image_patches = []
                for box in boxes:
                    patch = self._extract_image_patch(image, box, image_shape[:2])
                    if patch is None:
                        patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
                    image_patches.append(patch)
                    
                image_patches = np.asarray(image_patches)
                features = session.run(output_var, feed_dict={input_var: image_patches})
                return features
                
            return encoder
            
        except Exception as e:
            print(f"警告: 无法加载TensorFlow特征提取模型: {e}")
            print("使用备用特征提取方法 (仅使用边界框特征)")
            return self._fallback_encoder
            
    def _extract_image_patch(self, image, bbox, patch_shape):
        """
        从图像中提取边界框区域
        
        Args:
            image: 输入图像
            bbox: 边界框 (x, y, w, h)
            patch_shape: 目标patch尺寸
            
        Returns:
            图像patch
        """
        bbox = np.array(bbox)
        if patch_shape is not None:
            # 调整宽高比
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width
            
        # 转换为左上角和右下角坐标
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int64)
        
        # 裁剪到图像边界
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        
        if np.any(bbox[:2] >= bbox[2:]):
            return None
            
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image
        
    def _fallback_encoder(self, image, boxes):
        """
        备用特征提取方法 (当TensorFlow不可用时)
        使用简单的边界框特征
        
        Args:
            image: 输入图像
            boxes: 边界框列表
            
        Returns:
            特征向量数组
        """
        features = []
        img_h, img_w = image.shape[:2]
        
        for box in boxes:
            # 使用归一化的边界框坐标作为特征
            x, y, w, h = box
            feature = np.array([
                x / img_w,
                y / img_h,
                w / img_w,
                h / img_h,
                (x + w/2) / img_w,  # 中心点x
                (y + h/2) / img_h,  # 中心点y
                w * h / (img_w * img_h),  # 面积比
            ], dtype=np.float32)
            
            # 扩展到128维特征
            feature = np.tile(feature, 19)[:128]
            features.append(feature)
            
        return np.array(features)
        
    def update(self, frame: np.ndarray, detections: list) -> list:
        """
        更新跟踪器
        
        Args:
            frame: 当前帧图像 (BGR格式)
            detections: 检测结果列表 [[x1, y1, x2, y2, conf, class_id, class_name], ...]
            
        Returns:
            跟踪结果列表 [{'track_id': int, 'bbox': [x1,y1,x2,y2], 'class_id': int, 'class_name': str, 'confidence': float}, ...]
        """
        # 转换检测结果为DeepSORT格式
        if not detections:
            # 没有检测结果时，仍需更新跟踪器
            self.tracker.predict()
            self.tracker.update([])
            self._update_tracks()
            return self.tracks
            
        # 提取边界框和置信度
        bboxes = []
        confidences = []
        class_ids = []
        class_names = []
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_id, cls_name = det[:7]
            # 转换为tlwh格式
            w, h = x2 - x1, y2 - y1
            bboxes.append([x1, y1, w, h])
            confidences.append(conf)
            class_ids.append(int(cls_id))
            class_names.append(cls_name)
            
        bboxes = np.array(bboxes)
        confidences = np.array(confidences)
        
        # 提取特征
        features = self.encoder(frame, bboxes)
        
        # 创建Detection对象
        detection_objects = [
            Detection(bbox, confidence, feature)
            for bbox, confidence, feature in zip(bboxes, confidences, features)
        ]
        
        # 更新跟踪器
        self.tracker.predict()
        self.tracker.update(detection_objects)
        
        # 更新跟踪结果
        self._update_tracks(class_ids, class_names, confidences, bboxes)
        
        return self.tracks
        
    def _update_tracks(self, class_ids=None, class_names=None, confidences=None, bboxes=None):
        """
        更新跟踪结果
        
        Args:
            class_ids: 检测类别ID列表
            class_names: 检测类别名称列表
            confidences: 检测置信度列表
            bboxes: 检测边界框列表
        """
        self.tracks = []
        
        # 创建检测索引映射 (用于关联类别信息)
        det_idx_map = {}
        if bboxes is not None:
            for i, bbox in enumerate(bboxes):
                # 使用边界框中心点作为key
                key = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
                det_idx_map[key] = i
        
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
                
            # 获取边界框 (tlwh格式)
            tlwh = track.to_tlwh()
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            
            # 尝试匹配检测结果的类别信息
            class_id = -1
            class_name = "unknown"
            confidence = 0.0
            
            # 通过边界框中心点匹配
            center = (int(x1 + w/2), int(y1 + h/2))
            for key, idx in det_idx_map.items():
                # 检查中心点是否接近
                if abs(key[0] - center[0]) < max(w/2, 10) and abs(key[1] - center[1]) < max(h/2, 10):
                    if class_ids is not None and idx < len(class_ids):
                        class_id = class_ids[idx]
                    if class_names is not None and idx < len(class_names):
                        class_name = class_names[idx]
                    if confidences is not None and idx < len(confidences):
                        confidence = confidences[idx]
                    break
            
            self.tracks.append({
                'track_id': track.track_id,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(confidence),
            })
            
    def reset(self):
        """重置跟踪器"""
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        self.tracker = Tracker(metric)
        self.tracks = []
        
    def get_active_tracks(self) -> list:
        """
        获取当前活跃的跟踪目标
        
        Returns:
            跟踪结果列表
        """
        return self.tracks
        
    def get_track_count(self) -> dict:
        """
        获取各类别的跟踪数量统计
        
        Returns:
            类别数量统计字典
        """
        counts = {}
        for track in self.tracks:
            cls_name = track['class_name']
            counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts
