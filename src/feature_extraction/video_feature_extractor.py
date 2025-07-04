"""
视频特征提取器模块

提取视频数据的各种特征，包括视觉特征、动作特征、情感特征等
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import torch

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# 添加ProtoBully项目目录到路径
protobully_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(protobully_root)

from src.feature_extraction.base_feature_extractor import BaseFeatureExtractor
from src.utils.config import DATA_CONFIG, PREPROCESSING_CONFIG, FEATURE_CONFIG
from src.utils.utils import ensure_dir, save_json, load_json, save_pickle, load_pickle, get_timestamp

class VideoFeatureExtractor(BaseFeatureExtractor):
    """
    视频特征提取器类
    
    提取视频数据的各种特征，包括视觉特征、动作特征、情感特征等
    """
    
    def __init__(self, config=None, preprocessing_config=None, feature_config=None):
        """
        初始化视频特征提取器
        
        参数:
            config: 数据配置，如果为None则使用默认配置
            preprocessing_config: 预处理配置，如果为None则使用默认配置
            feature_config: 特征提取配置，如果为None则使用默认配置
        """
        super().__init__(config, preprocessing_config, feature_config)
        
        # 初始化视频特征提取模型
        self.visual_model = None
        self.action_model = None
        self.emotion_model = None
        
        # 初始化模型
        self._initialize_models()
        
        # 特征存储
        self.video_features = {}
        self.frame_features = {}
        self.segment_features = {}
    
    def _initialize_models(self):
        """初始化视频特征提取模型"""
        # 尝试加载视觉特征提取模型（例如ResNet）
        try:
            import torchvision.models as models
            
            model_name = self.feature_config.get("video_model", "resnet50")
            self.logger.info(f"加载视觉特征提取模型: {model_name}")
            
            if model_name == "resnet50":
                self.visual_model = models.resnet50(pretrained=True)
                # 移除最后的全连接层，只保留特征提取部分
                self.visual_model = torch.nn.Sequential(*list(self.visual_model.children())[:-1])
            elif model_name == "resnet18":
                self.visual_model = models.resnet18(pretrained=True)
                self.visual_model = torch.nn.Sequential(*list(self.visual_model.children())[:-1])
            else:
                self.logger.warning(f"不支持的视觉模型: {model_name}，将使用ResNet50")
                self.visual_model = models.resnet50(pretrained=True)
                self.visual_model = torch.nn.Sequential(*list(self.visual_model.children())[:-1])
            
            # 如果有GPU，将模型移到GPU
            if torch.cuda.is_available():
                self.visual_model = self.visual_model.cuda()
                self.logger.info("视觉模型已移至GPU")
            
            self.visual_model.eval()  # 设置为评估模式
            self.logger.info("视觉特征提取模型加载成功")
            
        except Exception as e:
            self.logger.warning(f"加载视觉特征提取模型失败: {str(e)}")
            self.visual_model = None
        
        # 尝试加载动作识别模型（如果需要）
        try:
            # 此处可以加载专门的动作识别模型，如I3D或SlowFast
            # 为简化起见，我们暂时不实现动作识别
            self.action_model = None
            self.logger.info("动作识别模型未加载")
            
        except Exception as e:
            self.logger.warning(f"加载动作识别模型失败: {str(e)}")
            self.action_model = None
        
        # 尝试加载表情识别模型（如果需要）
        try:
            # 此处可以加载专门的表情识别模型
            # 为简化起见，我们暂时不实现表情识别
            self.emotion_model = None
            self.logger.info("表情识别模型未加载")
            
        except Exception as e:
            self.logger.warning(f"加载表情识别模型失败: {str(e)}")
            self.emotion_model = None
    
    def extract_visual_features(self, video_frames):
        """
        提取视觉特征
        
        参数:
            video_frames: 视频帧列表，每帧为张量形状[C, H, W]
            
        返回:
            视觉特征张量，形状为[num_frames, feature_dim]
        """
        if not self.visual_model or not video_frames:
            return None
        
        self.logger.info(f"为 {len(video_frames)} 帧提取视觉特征")
        
        # 批处理大小
        batch_size = 16
        
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(video_frames), batch_size), desc="提取视觉特征"):
                batch_frames = video_frames[i:i+batch_size]
                
                # 将列表转换为批次张量
                batch_tensor = torch.stack(batch_frames)
                
                # 如果有GPU，将输入移到GPU
                if torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda()
                
                # 获取特征
                batch_features = self.visual_model(batch_tensor).squeeze(-1).squeeze(-1)
                
                # 将特征移回CPU
                batch_features = batch_features.cpu().numpy()
                
                features.append(batch_features)
        
        # 合并所有批次的特征
        all_features = np.vstack(features)
        
        self.logger.info(f"视觉特征提取完成，形状: {all_features.shape}")
        return all_features
    
    def extract_action_features(self, video_clips):
        """
        提取动作特征（如果模型可用）
        
        参数:
            video_clips: 视频片段列表
            
        返回:
            动作特征
        """
        # 此方法为占位符，实际实现可能需要特定的动作识别模型
        self.logger.warning("动作特征提取未实现")
        return None
    
    def extract_emotion_features(self, face_frames):
        """
        提取表情特征（如果模型可用）
        
        参数:
            face_frames: 人脸帧列表
            
        返回:
            表情特征
        """
        # 此方法为占位符，实际实现可能需要特定的表情识别模型
        self.logger.warning("表情特征提取未实现")
        return None
    
    def extract_scene_features(self, video_data):
        """
        提取场景特征
        
        参数:
            video_data: 视频数据
            
        返回:
            场景特征
        """
        scene_features = {}
        
        for post_id, post_data in tqdm(video_data.items(), desc="提取场景特征"):
            # 获取场景注释
            annotations = post_data.get('annotations', {})
            
            # 提取场景标签
            scene_labels = annotations.get('scene_labels', [])
            
            # 场景类型统计
            scene_types = {}
            for label in scene_labels:
                scene_type = label.get('label', '')
                if scene_type:
                    if scene_type not in scene_types:
                        scene_types[scene_type] = 0
                    scene_types[scene_type] += 1
            
            # 主要场景类型
            main_scene_type = max(scene_types.items(), key=lambda x: x[1])[0] if scene_types else ""
            
            # 场景变化次数
            scene_changes = len(scene_labels) - 1 if len(scene_labels) > 0 else 0
            
            # 场景持续时间
            scene_durations = []
            for i in range(len(scene_labels)):
                if i < len(scene_labels) - 1:
                    start_time = scene_labels[i].get('start_time', 0)
                    end_time = scene_labels[i+1].get('start_time', 0)
                    duration = end_time - start_time
                    if duration > 0:
                        scene_durations.append(duration)
            
            avg_scene_duration = np.mean(scene_durations) if scene_durations else 0
            
            # 存储场景特征
            scene_features[post_id] = {
                'scene_types': scene_types,
                'main_scene_type': main_scene_type,
                'scene_changes': scene_changes,
                'avg_scene_duration': avg_scene_duration
            }
        
        return scene_features
    
    def extract_features(self, data):
        """
        提取视频特征
        
        参数:
            data: 视频数据
            
        返回:
            提取的特征
        """
        self.logger.info("开始提取视频特征")
        
        # 初始化特征存储
        video_features = {}
        frame_features = {}
        segment_features = {}
        
        # 提取场景特征
        scene_features = self.extract_scene_features(data)
        
        # 处理每个视频
        for post_id, post_data in tqdm(data.items(), desc="提取视频特征"):
            frames = post_data.get('frames', [])
            annotations = post_data.get('annotations', {})
            label = post_data.get('label', 0)
            
            # 跳过没有帧的视频
            if not frames:
                continue
            
            # 从帧中提取特征
            # 注意：实际应用中，frames应该是预处理好的张量
            # 这里我们假设frames已经是张量列表
            
            # 视觉特征
            visual_features = None
            if self.visual_model and isinstance(frames[0], torch.Tensor):
                visual_features = self.extract_visual_features(frames)
            
            # 统计视频信息
            num_frames = len(frames)
            duration = post_data.get('duration', 0)
            fps = num_frames / duration if duration > 0 else 0
            
            # 视频级特征
            video_feature = {
                'post_id': post_id,
                'label': label,
                'num_frames': num_frames,
                'duration': duration,
                'fps': fps,
                'visual_features': visual_features.tolist() if visual_features is not None else None
            }
            
            # 添加场景特征
            if post_id in scene_features:
                video_feature['scene_features'] = scene_features[post_id]
            
            # 存储视频特征
            video_features[post_id] = video_feature
            
            # 处理帧级特征
            if visual_features is not None:
                for i, frame_feature in enumerate(visual_features):
                    frame_id = f"{post_id}_frame_{i}"
                    frame_features[frame_id] = {
                        'frame_id': frame_id,
                        'post_id': post_id,
                        'frame_index': i,
                        'features': frame_feature.tolist()
                    }
            
            # 处理片段级特征
            segments = annotations.get('segments', [])
            for segment in segments:
                segment_id = segment.get('segment_id', '')
                if not segment_id:
                    continue
                
                start_frame = segment.get('start_frame', 0)
                end_frame = segment.get('end_frame', 0)
                
                # 如果有视觉特征，计算片段的平均特征
                segment_visual_features = None
                if visual_features is not None and start_frame < end_frame <= len(visual_features):
                    segment_visual_features = np.mean(visual_features[start_frame:end_frame], axis=0)
                
                segment_features[segment_id] = {
                    'segment_id': segment_id,
                    'post_id': post_id,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'duration_frames': end_frame - start_frame,
                    'features': segment_visual_features.tolist() if segment_visual_features is not None else None
                }
        
        self.logger.info(f"提取了 {len(video_features)} 个视频的特征")
        self.logger.info(f"提取了 {len(frame_features)} 帧的特征")
        self.logger.info(f"提取了 {len(segment_features)} 个片段的特征")
        
        # 存储特征
        self.video_features = video_features
        self.frame_features = frame_features
        self.segment_features = segment_features
        
        return {
            'video_features': video_features,
            'frame_features': frame_features,
            'segment_features': segment_features
        }
    
    def get_feature_names(self):
        """
        获取特征名称列表
        
        返回:
            特征名称列表
        """
        feature_names = [
            # 基础视频特征
            'num_frames',
            'duration',
            'fps',
            
            # 场景特征
            'scene_changes',
            'avg_scene_duration',
            
            # 视觉特征
            # 注意：视觉特征的维度可能很高，这里只是示例
            'visual_feature_mean',
            'visual_feature_std'
        ]
        
        return feature_names
    
    def get_processed_features(self, data=None, force_reprocess=False):
        """
        获取处理后的特征，如果已存在则加载，否则提取
        
        参数:
            data: 要提取特征的数据，如果为None则尝试加载
            force_reprocess: 是否强制重新处理
            
        返回:
            处理后的特征
        """
        # 特征文件路径
        features_file = os.path.join(self.features_path, 'video_features.pkl')
        
        # 如果不强制重新处理且特征文件存在，则加载
        if not force_reprocess and os.path.exists(features_file):
            self.logger.info(f"从 {features_file} 加载已处理的视频特征")
            features = self.load_features(features_file)
            
            # 更新实例变量
            self.video_features = features.get('video_features', {})
            self.frame_features = features.get('frame_features', {})
            self.segment_features = features.get('segment_features', {})
            
            return features
        
        # 如果需要重新处理但未提供数据，则加载数据
        if data is None:
            video_data_file = os.path.join(self.config["video_data_path"], 'processed_video_data.json')
            self.logger.info(f"从 {video_data_file} 加载视频数据")
            data = self.load_processed_data(video_data_file)
        
        # 提取特征
        features = self.extract_features(data)
        
        # 保存特征
        self.save_features(features, features_file)
        
        return features

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="视频特征提取")
    parser.add_argument("--force", action="store_true", help="强制重新处理")
    args = parser.parse_args()
    
    extractor = VideoFeatureExtractor()
    features = extractor.get_processed_features(force_reprocess=args.force)
    
    print(f"提取了 {len(features['video_features'])} 个视频的特征")
    print(f"提取了 {len(features['frame_features'])} 帧的特征")
    print(f"提取了 {len(features['segment_features'])} 个片段的特征")

if __name__ == "__main__":
    main() 