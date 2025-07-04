"""
多模态特征提取器模块

集成各种模态特征并进行特征融合
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# 添加ProtoBully项目目录到路径
protobully_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(protobully_root)

from src.feature_extraction.base_feature_extractor import BaseFeatureExtractor
from src.feature_extraction.text_feature_extractor import TextFeatureExtractor
from src.feature_extraction.video_feature_extractor import VideoFeatureExtractor
from src.feature_extraction.user_feature_extractor import UserFeatureExtractor
from src.utils.config import DATA_CONFIG, PREPROCESSING_CONFIG, FEATURE_CONFIG, ALIGNMENT_CONFIG
from src.utils.utils import ensure_dir, save_json, load_json, save_pickle, load_pickle, get_timestamp

class MultimodalFeatureExtractor(BaseFeatureExtractor):
    """
    多模态特征提取器类
    
    集成各种模态特征并进行特征融合
    """
    
    def __init__(self, config=None, preprocessing_config=None, feature_config=None, alignment_config=None):
        """
        初始化多模态特征提取器
        
        参数:
            config: 数据配置，如果为None则使用默认配置
            preprocessing_config: 预处理配置，如果为None则使用默认配置
            feature_config: 特征提取配置，如果为None则使用默认配置
            alignment_config: 对齐配置，如果为None则使用默认配置
        """
        super().__init__(config, preprocessing_config, feature_config)
        
        # 初始化对齐配置
        self.alignment_config = alignment_config or ALIGNMENT_CONFIG
        
        # 初始化各模态特征提取器
        self.text_extractor = TextFeatureExtractor(config, preprocessing_config, feature_config)
        self.video_extractor = VideoFeatureExtractor(config, preprocessing_config, feature_config)
        self.user_extractor = UserFeatureExtractor(config, preprocessing_config, feature_config)
        
        # 特征存储
        self.multimodal_features = {}
        self.feature_names = []
        self.dataset_splits = {}
    
    def load_modality_features(self, force_reprocess=False):
        """
        加载各模态特征
        
        参数:
            force_reprocess: 是否强制重新处理
            
        返回:
            各模态特征的字典
        """
        self.logger.info("加载各模态特征")
        
        # 加载文本特征
        text_features = self.text_extractor.get_processed_features(force_reprocess=force_reprocess)
        self.logger.info(f"加载了 {len(text_features['session_features'])} 个会话的文本特征")
        
        # 加载视频特征
        video_features = self.video_extractor.get_processed_features(force_reprocess=force_reprocess)
        self.logger.info(f"加载了 {len(video_features['video_features'])} 个会话的视频特征")
        
        # 加载用户特征
        user_features = self.user_extractor.get_processed_features(force_reprocess=force_reprocess)
        self.logger.info(f"加载了 {len(user_features['session_user_features'])} 个会话的用户特征")
        
        return {
            'text': text_features,
            'video': video_features,
            'user': user_features
        }
    
    def align_modalities(self, modality_features):
        """
        对齐各模态特征
        
        参数:
            modality_features: 各模态特征的字典
            
        返回:
            对齐后的特征
        """
        self.logger.info("对齐各模态特征")
        
        # 获取各模态的会话特征
        text_session_features = modality_features.get('text', {}).get('session_features', {})
        video_session_features = modality_features.get('video', {}).get('video_features', {})
        user_session_features = modality_features.get('user', {}).get('session_user_features', {})
        
        # 检查是否所有模态都存在数据
        if not text_session_features:
            self.logger.warning("文本模态数据为空")
        if not video_session_features:
            self.logger.warning("视频模态数据为空")
        if not user_session_features:
            self.logger.warning("用户模态数据为空")
        
        # 获取所有会话ID
        all_post_ids = set(text_session_features.keys()) | set(video_session_features.keys()) | set(user_session_features.keys())
        
        # 如果存在至少两个模态的数据，则使用这些数据进行对齐
        # 否则，使用单一模态的数据
        if len(text_session_features) > 0 and len(video_session_features) > 0:
            common_post_ids = set(text_session_features.keys()) & set(video_session_features.keys())
            self.logger.info(f"文本-视频模态共有 {len(common_post_ids)} 个会话")
        elif len(text_session_features) > 0 and len(user_session_features) > 0:
            common_post_ids = set(text_session_features.keys()) & set(user_session_features.keys())
            self.logger.info(f"文本-用户模态共有 {len(common_post_ids)} 个会话")
        elif len(video_session_features) > 0 and len(user_session_features) > 0:
            common_post_ids = set(video_session_features.keys()) & set(user_session_features.keys())
            self.logger.info(f"视频-用户模态共有 {len(common_post_ids)} 个会话")
        else:
            # 如果只有一个模态有数据，使用该模态的所有会话ID
            if len(text_session_features) > 0:
                common_post_ids = set(text_session_features.keys())
                self.logger.info(f"仅使用文本模态数据，共 {len(common_post_ids)} 个会话")
            elif len(video_session_features) > 0:
                common_post_ids = set(video_session_features.keys())
                self.logger.info(f"仅使用视频模态数据，共 {len(common_post_ids)} 个会话")
            elif len(user_session_features) > 0:
                common_post_ids = set(user_session_features.keys())
                self.logger.info(f"仅使用用户模态数据，共 {len(common_post_ids)} 个会话")
            else:
                common_post_ids = set()
                self.logger.warning("所有模态数据都为空，无法进行对齐")
        
        self.logger.info(f"共有 {len(all_post_ids)} 个会话，其中 {len(common_post_ids)} 个会话将用于特征提取")
        
        # 对齐后的特征
        aligned_features = {}
        
        # 处理每个会话
        for post_id in tqdm(common_post_ids, desc="对齐特征"):
            # 获取各模态特征
            text_feature = text_session_features.get(post_id, {})
            video_feature = video_session_features.get(post_id, {})
            user_feature = user_session_features.get(post_id, {})
            
            # 获取标签 (优先使用文本模态的标签)
            label = text_feature.get('label', user_feature.get('label', video_feature.get('label', 0)))
            
            # 创建对齐特征
            aligned_feature = {
                'post_id': post_id,
                'label': label,
                'text_features': {},
                'video_features': {},
                'user_features': {}
            }
            
            # 添加文本特征
            if text_feature:
                # 基本特征
                aligned_feature['text_features']['sentiment_compound'] = text_feature.get('avg_sentiment_compound', 0)
                aligned_feature['text_features']['offensive_score'] = text_feature.get('avg_offensive_score', 0)
                aligned_feature['text_features']['max_offensive_score'] = text_feature.get('max_offensive_score', 0)
                
                # 统计特征
                stat_features = text_feature.get('statistical_features', {})
                for key, value in stat_features.items():
                    aligned_feature['text_features'][key] = value
                
                # BERT嵌入
                if 'text_embeddings' in text_feature and text_feature['text_embeddings'] is not None:
                    aligned_feature['text_features']['embeddings'] = text_feature['text_embeddings']
            
            # 添加视频特征
            if video_feature:
                # 基本特征
                aligned_feature['video_features']['num_frames'] = video_feature.get('num_frames', 0)
                aligned_feature['video_features']['duration'] = video_feature.get('duration', 0)
                aligned_feature['video_features']['fps'] = video_feature.get('fps', 0)
                
                # 场景特征
                scene_features = video_feature.get('scene_features', {})
                for key, value in scene_features.items():
                    if key != 'scene_types':  # 跳过复杂字典
                        aligned_feature['video_features'][f'scene_{key}'] = value
                
                # 视觉特征
                if 'visual_features' in video_feature and video_feature['visual_features'] is not None:
                    visual_features = np.array(video_feature['visual_features'])
                    aligned_feature['video_features']['visual_mean'] = np.mean(visual_features, axis=0).tolist()
                    aligned_feature['video_features']['visual_std'] = np.std(visual_features, axis=0).tolist()
            
            # 添加用户特征
            if user_feature:
                # 基本特征
                aligned_feature['user_features']['user_count'] = user_feature.get('user_count', 0)
                aligned_feature['user_features']['avg_user_degree'] = user_feature.get('avg_user_degree', 0)
                aligned_feature['user_features']['max_user_degree'] = user_feature.get('max_user_degree', 0)
                aligned_feature['user_features']['avg_user_pagerank'] = user_feature.get('avg_user_pagerank', 0)
                aligned_feature['user_features']['max_user_pagerank'] = user_feature.get('max_user_pagerank', 0)
                aligned_feature['user_features']['avg_bullying_ratio'] = user_feature.get('avg_bullying_ratio', 0)
                aligned_feature['user_features']['max_bullying_ratio'] = user_feature.get('max_bullying_ratio', 0)
                
                # 时间特征
                temporal_features = user_feature.get('temporal_features', {})
                for key, value in temporal_features.items():
                    aligned_feature['user_features'][key] = value
            
            # 添加到对齐特征字典
            aligned_features[post_id] = aligned_feature
        
        self.logger.info(f"特征对齐完成，对齐了 {len(aligned_features)} 个会话的特征")
        
        return aligned_features
    
    def extract_cross_modal_features(self, aligned_features):
        """
        提取跨模态特征
        
        参数:
            aligned_features: 对齐后的特征
            
        返回:
            添加了跨模态特征的特征
        """
        self.logger.info("提取跨模态特征")
        
        # 处理每个会话
        for post_id, feature in tqdm(aligned_features.items(), desc="提取跨模态特征"):
            # 初始化跨模态特征
            feature['cross_modal_features'] = {}
            
            # 计算文本-视频对齐
            if 'text_features' in feature and 'video_features' in feature:
                # 情感与场景的关联
                if 'sentiment_compound' in feature['text_features'] and 'scene_main_scene_type' in feature['video_features']:
                    sentiment = feature['text_features']['sentiment_compound']
                    scene_type = feature['video_features']['scene_main_scene_type']
                    
                    feature['cross_modal_features']['text_video_sentiment_scene'] = {
                        'sentiment': sentiment,
                        'scene_type': scene_type
                    }
                
                # 攻击性与视频时长的关联
                if 'offensive_score' in feature['text_features'] and 'duration' in feature['video_features']:
                    offensive_score = feature['text_features']['offensive_score']
                    duration = feature['video_features']['duration']
                    
                    feature['cross_modal_features']['text_video_offensive_duration'] = offensive_score * duration
            
            # 计算文本-用户对齐
            if 'text_features' in feature and 'user_features' in feature:
                # 攻击性与用户霸凌比例的关联
                if 'offensive_score' in feature['text_features'] and 'avg_bullying_ratio' in feature['user_features']:
                    offensive_score = feature['text_features']['offensive_score']
                    bullying_ratio = feature['user_features']['avg_bullying_ratio']
                    
                    feature['cross_modal_features']['text_user_offensive_bullying'] = offensive_score * bullying_ratio
                
                # 情感与用户网络影响力的关联
                if 'sentiment_compound' in feature['text_features'] and 'max_user_pagerank' in feature['user_features']:
                    sentiment = feature['text_features']['sentiment_compound']
                    pagerank = feature['user_features']['max_user_pagerank']
                    
                    feature['cross_modal_features']['text_user_sentiment_influence'] = sentiment * pagerank
            
            # 计算视频-用户对齐
            if 'video_features' in feature and 'user_features' in feature:
                # 视频时长与用户数量的关联
                if 'duration' in feature['video_features'] and 'user_count' in feature['user_features']:
                    duration = feature['video_features']['duration']
                    user_count = feature['user_features']['user_count']
                    
                    feature['cross_modal_features']['video_user_duration_count'] = duration / user_count if user_count > 0 else 0
        
        self.logger.info("跨模态特征提取完成")
        
        return aligned_features
    
    def create_feature_vectors(self, aligned_features):
        """
        创建特征向量
        
        参数:
            aligned_features: 对齐后的特征
            
        返回:
            特征向量和标签
        """
        self.logger.info("创建特征向量")
        
        # 收集所有特征名称
        feature_names = []
        
        # 根据第一个会话收集特征名称
        first_post_id = next(iter(aligned_features))
        first_feature = aligned_features[first_post_id]
        
        # 收集文本特征名称
        if 'text_features' in first_feature:
            for key in first_feature['text_features']:
                if key != 'embeddings':  # 嵌入单独处理
                    feature_names.append(f'text_{key}')
        
        # 收集视频特征名称
        if 'video_features' in first_feature:
            for key in first_feature['video_features']:
                if key not in ['visual_mean', 'visual_std']:  # 视觉特征单独处理
                    feature_names.append(f'video_{key}')
        
        # 收集用户特征名称
        if 'user_features' in first_feature:
            for key in first_feature['user_features']:
                feature_names.append(f'user_{key}')
        
        # 收集跨模态特征名称
        if 'cross_modal_features' in first_feature:
            for key, value in first_feature['cross_modal_features'].items():
                if isinstance(value, dict):
                    for subkey in value:
                        feature_names.append(f'cross_{key}_{subkey}')
                else:
                    feature_names.append(f'cross_{key}')
        
        # 添加嵌入维度（如果有）
        has_text_embeddings = ('text_features' in first_feature and 
                               'embeddings' in first_feature['text_features'] and 
                               first_feature['text_features']['embeddings'] is not None)
        
        if has_text_embeddings:
            embedding_dim = len(first_feature['text_features']['embeddings'])
            for i in range(embedding_dim):
                feature_names.append(f'text_embedding_{i}')
        
        has_visual_features = ('video_features' in first_feature and 
                              'visual_mean' in first_feature['video_features'] and 
                              first_feature['video_features']['visual_mean'] is not None)
        
        if has_visual_features:
            visual_dim = len(first_feature['video_features']['visual_mean'])
            for i in range(visual_dim):
                feature_names.append(f'video_visual_mean_{i}')
                feature_names.append(f'video_visual_std_{i}')
        
        self.logger.info(f"共有 {len(feature_names)} 个特征")
        
        # 创建特征矩阵和标签
        feature_matrix = []
        labels = []
        post_ids = []
        
        # 处理每个会话
        for post_id, feature in tqdm(aligned_features.items(), desc="构建特征向量"):
            feature_vector = []
            
            # 添加文本特征
            if 'text_features' in feature:
                for key in feature['text_features']:
                    if key != 'embeddings':
                        feature_vector.append(feature['text_features'][key])
            
            # 添加视频特征
            if 'video_features' in feature:
                for key in feature['video_features']:
                    if key not in ['visual_mean', 'visual_std']:
                        feature_vector.append(feature['video_features'][key])
            
            # 添加用户特征
            if 'user_features' in feature:
                for key in feature['user_features']:
                    feature_vector.append(feature['user_features'][key])
            
            # 添加跨模态特征
            if 'cross_modal_features' in feature:
                for key, value in feature['cross_modal_features'].items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            feature_vector.append(subvalue)
                    else:
                        feature_vector.append(value)
            
            # 添加文本嵌入
            if has_text_embeddings and 'text_features' in feature and 'embeddings' in feature['text_features']:
                embeddings = feature['text_features']['embeddings']
                feature_vector.extend(embeddings)
            
            # 添加视觉特征
            if has_visual_features and 'video_features' in feature:
                if 'visual_mean' in feature['video_features'] and 'visual_std' in feature['video_features']:
                    visual_mean = feature['video_features']['visual_mean']
                    visual_std = feature['video_features']['visual_std']
                    feature_vector.extend(visual_mean)
                    feature_vector.extend(visual_std)
            
            # 添加到特征矩阵
            feature_matrix.append(feature_vector)
            
            # 添加标签
            labels.append(feature.get('label', 0))
            
            # 添加会话ID
            post_ids.append(post_id)
        
        # 转换为NumPy数组
        feature_matrix = np.array(feature_matrix)
        labels = np.array(labels)
        
        self.logger.info(f"特征向量创建完成，形状: {feature_matrix.shape}")
        
        # 保存特征名称
        self.feature_names = feature_names
        
        return feature_matrix, labels, post_ids
    
    def normalize_feature_matrix(self, feature_matrix):
        """
        标准化特征矩阵
        
        参数:
            feature_matrix: 特征矩阵
            
        返回:
            标准化后的特征矩阵和标准化器
        """
        self.logger.info("标准化特征矩阵")
        
        # 创建标准化器
        scaler = StandardScaler()
        
        # 标准化特征
        normalized_matrix = scaler.fit_transform(feature_matrix)
        
        self.logger.info("特征标准化完成")
        
        return normalized_matrix, scaler
    
    def split_dataset(self, post_ids, labels):
        """
        划分数据集
        
        参数:
            post_ids: 会话ID列表
            labels: 标签列表
            
        返回:
            数据集划分
        """
        self.logger.info("划分数据集")
        
        # 从多模态处理器加载数据集划分
        try:
            multimodal_data_file = os.path.join(self.config["aligned_data_path"], 'multimodal_data.json')
            multimodal_data = self.load_processed_data(multimodal_data_file)
            
            dataset_splits = multimodal_data.get('dataset_splits', {})
            
            if dataset_splits and 'train' in dataset_splits and 'val' in dataset_splits and 'test' in dataset_splits:
                self.logger.info("使用现有的数据集划分")
                
                # 确保所有post_id都在我们的数据中
                train_ids = [pid for pid in dataset_splits['train'] if pid in post_ids]
                val_ids = [pid for pid in dataset_splits['val'] if pid in post_ids]
                test_ids = [pid for pid in dataset_splits['test'] if pid in post_ids]
                
                dataset_splits = {
                    'train': train_ids,
                    'val': val_ids,
                    'test': test_ids
                }
                
                self.logger.info(f"训练集: {len(train_ids)}，验证集: {len(val_ids)}，测试集: {len(test_ids)}")
                
                return dataset_splits
        except Exception as e:
            self.logger.warning(f"加载现有数据集划分失败: {str(e)}")
        
        # 手动划分数据集
        from sklearn.model_selection import train_test_split
        
        # 转换为NumPy数组
        post_ids = np.array(post_ids)
        labels = np.array(labels)
        
        # 先划分训练集和临时集
        train_ratio = self.preprocessing_config.get("train_test_split_ratio", 0.7)
        train_indices, temp_indices = train_test_split(
            np.arange(len(post_ids)),
            test_size=1 - train_ratio,
            random_state=self.preprocessing_config.get("random_seed", 42),
            stratify=labels
        )
        
        # 从临时集划分验证集和测试集
        val_ratio = self.preprocessing_config.get("val_test_split_ratio", 0.5)
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=1 - val_ratio,
            random_state=self.preprocessing_config.get("random_seed", 42),
            stratify=labels[temp_indices]
        )
        
        # 获取对应的会话ID
        train_ids = post_ids[train_indices].tolist()
        val_ids = post_ids[val_indices].tolist()
        test_ids = post_ids[test_indices].tolist()
        
        dataset_splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        
        self.logger.info(f"训练集: {len(train_ids)}，验证集: {len(val_ids)}，测试集: {len(test_ids)}")
        
        return dataset_splits
    
    def extract_features(self, data=None):
        """
        提取多模态特征
        
        参数:
            data: 数据，此参数在这里未使用，保留是为了与基类接口一致
            
        返回:
            提取的特征
        """
        self.logger.info("开始提取多模态特征")
        
        # 加载各模态特征
        modality_features = self.load_modality_features()
        
        # 对齐各模态特征
        aligned_features = self.align_modalities(modality_features)
        
        # 提取跨模态特征
        aligned_features = self.extract_cross_modal_features(aligned_features)
        
        # 创建特征向量
        feature_matrix, labels, post_ids = self.create_feature_vectors(aligned_features)
        
        # 标准化特征矩阵
        normalized_matrix, scaler = self.normalize_feature_matrix(feature_matrix)
        
        # 划分数据集
        dataset_splits = self.split_dataset(post_ids, labels)
        
        # 构建特征字典
        features = {
            'aligned_features': aligned_features,
            'feature_matrix': feature_matrix,
            'normalized_matrix': normalized_matrix,
            'labels': labels,
            'post_ids': post_ids,
            'feature_names': self.feature_names,
            'dataset_splits': dataset_splits,
            'scaler': scaler
        }
        
        # 保存特征
        self.multimodal_features = features
        self.dataset_splits = dataset_splits
        
        self.logger.info("多模态特征提取完成")
        
        return features
    
    def get_feature_names(self):
        """
        获取特征名称列表
        
        返回:
            特征名称列表
        """
        return self.feature_names
    
    def get_processed_features(self, data=None, force_reprocess=False):
        """
        获取处理后的特征，如果已存在则加载，否则提取
        
        参数:
            data: 要提取特征的数据，此参数在这里未使用，保留是为了与基类接口一致
            force_reprocess: 是否强制重新处理
            
        返回:
            处理后的特征
        """
        # 特征文件路径
        features_file = os.path.join(self.features_path, 'multimodal_features.pkl')
        
        # 如果不强制重新处理且特征文件存在，则加载
        if not force_reprocess and os.path.exists(features_file):
            self.logger.info(f"从 {features_file} 加载已处理的多模态特征")
            features = self.load_features(features_file)
            
            # 更新实例变量
            self.multimodal_features = features
            self.feature_names = features.get('feature_names', [])
            self.dataset_splits = features.get('dataset_splits', {})
            
            return features
        
        # 提取特征
        features = self.extract_features()
        
        # 保存特征
        self.save_features(features, features_file)
        
        return features

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多模态特征提取")
    parser.add_argument("--force", action="store_true", help="强制重新处理")
    args = parser.parse_args()
    
    extractor = MultimodalFeatureExtractor()
    features = extractor.get_processed_features(force_reprocess=args.force)
    
    print(f"提取了 {len(features['post_ids'])} 个会话的多模态特征")
    print(f"特征维度: {len(features['feature_names'])}")
    print(f"特征矩阵形状: {features['feature_matrix'].shape}")
    
    # 打印数据集划分
    dataset_splits = features['dataset_splits']
    print(f"训练集: {len(dataset_splits['train'])} 个会话")
    print(f"验证集: {len(dataset_splits['val'])} 个会话")
    print(f"测试集: {len(dataset_splits['test'])} 个会话")

if __name__ == "__main__":
    main() 