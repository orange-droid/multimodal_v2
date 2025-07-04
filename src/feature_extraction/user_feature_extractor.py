"""
用户特征提取器模块

提取用户数据的各种特征，包括活动特征、社交网络特征等
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
import networkx as nx

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

class UserFeatureExtractor(BaseFeatureExtractor):
    """
    用户特征提取器类
    
    提取用户数据的各种特征，包括活动特征、社交网络特征等
    """
    
    def __init__(self, config=None, preprocessing_config=None, feature_config=None):
        """
        初始化用户特征提取器
        
        参数:
            config: 数据配置，如果为None则使用默认配置
            preprocessing_config: 预处理配置，如果为None则使用默认配置
            feature_config: 特征提取配置，如果为None则使用默认配置
        """
        super().__init__(config, preprocessing_config, feature_config)
        
        # 特征存储
        self.user_features = {}
        self.social_network_features = {}
        self.temporal_features = {}
    
    def build_user_network(self, data):
        """
        构建用户交互网络
        
        参数:
            data: 元数据
            
        返回:
            用户交互网络
        """
        self.logger.info("构建用户交互网络")
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 统计用户之间的互动
        user_interactions = defaultdict(int)
        
        # 跟踪每个会话的用户参与情况
        session_users = defaultdict(set)
        
        # 处理每个会话
        for post_id, post_data in tqdm(data.items(), desc="分析用户交互"):
            comments = post_data.get('comments', [])
            
            # 映射评论到用户
            comment_to_user = {}
            
            # 收集会话中的所有用户
            for comment in comments:
                user_id = comment.get('userId', '')
                comment_id = comment.get('commentId', '')
                
                if user_id and comment_id:
                    comment_to_user[comment_id] = user_id
                    session_users[post_id].add(user_id)
                    
                    # 添加用户节点
                    if not G.has_node(user_id):
                        G.add_node(user_id)
            
            # 分析评论之间的回复关系
            for comment in comments:
                user_id = comment.get('userId', '')
                parent_comment_id = comment.get('parentCommentId', '')
                
                if user_id and parent_comment_id and parent_comment_id in comment_to_user:
                    parent_user_id = comment_to_user[parent_comment_id]
                    
                    if user_id != parent_user_id:  # 排除自回复
                        # 添加交互边
                        if G.has_edge(user_id, parent_user_id):
                            G[user_id][parent_user_id]['weight'] += 1
                        else:
                            G.add_edge(user_id, parent_user_id, weight=1)
                        
                        # 更新交互计数
                        user_interactions[(user_id, parent_user_id)] += 1
        
        self.logger.info(f"用户交互网络构建完成，包含 {G.number_of_nodes()} 个用户节点和 {G.number_of_edges()} 条交互边")
        
        return G, session_users
    
    def extract_network_features(self, G):
        """
        提取网络特征
        
        参数:
            G: 用户交互网络
            
        返回:
            网络特征
        """
        self.logger.info("提取网络特征")
        
        network_features = {}
        
        # 基本网络统计
        network_features['global'] = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
        }
        
        # 检查图是否为空，如果不为空才计算reciprocity
        if G.number_of_edges() > 0:
            try:
                network_features['global']['reciprocity'] = nx.reciprocity(G)
            except Exception as e:
                self.logger.warning(f"计算reciprocity失败: {str(e)}")
                network_features['global']['reciprocity'] = 0
        else:
            network_features['global']['reciprocity'] = 0
            self.logger.warning("图为空，reciprocity设置为0")
        
        # 尝试计算全局聚类系数（可能较慢）
        try:
            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                network_features['global']['clustering_coefficient'] = nx.average_clustering(G)
            else:
                network_features['global']['clustering_coefficient'] = 0
                self.logger.warning("图为空或无边，聚类系数设置为0")
        except Exception as e:
            self.logger.warning(f"计算聚类系数失败: {str(e)}")
            network_features['global']['clustering_coefficient'] = 0
        
        # 用户级网络特征
        user_network_features = {}
        
        # 如果图不为空，计算节点中心性度量
        if G.number_of_nodes() > 0:
            self.logger.info("计算节点中心性指标")
            
            # 出入度
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())
            
            # PageRank
            try:
                if G.number_of_edges() > 0:
                    pagerank = nx.pagerank(G)
                else:
                    pagerank = {node: 1.0/G.number_of_nodes() for node in G.nodes()}
                    self.logger.warning("图无边，PageRank设置为均匀分布")
            except Exception as e:
                self.logger.warning(f"计算PageRank失败: {str(e)}")
                pagerank = {node: 0 for node in G.nodes()}
            
            # 接近中心性
            try:
                if G.number_of_edges() > 0:
                    closeness = nx.closeness_centrality(G)
                else:
                    closeness = {node: 0 for node in G.nodes()}
                    self.logger.warning("图无边，接近中心性设置为0")
            except Exception as e:
                self.logger.warning(f"计算接近中心性失败: {str(e)}")
                closeness = {node: 0 for node in G.nodes()}
            
            # 介数中心性（计算较慢，可选）
            betweenness = None
            if G.number_of_nodes() < 1000 and G.number_of_edges() > 0:  # 只为小网络计算
                try:
                    betweenness = nx.betweenness_centrality(G)
                except Exception as e:
                    self.logger.warning(f"计算介数中心性失败: {str(e)}")
            
            # 组合节点特征
            for node in G.nodes():
                user_network_features[node] = {
                    'in_degree': in_degree.get(node, 0),
                    'out_degree': out_degree.get(node, 0),
                    'total_degree': in_degree.get(node, 0) + out_degree.get(node, 0),
                    'pagerank': pagerank.get(node, 0),
                    'closeness': closeness.get(node, 0),
                }
                
                if betweenness:
                    user_network_features[node]['betweenness'] = betweenness.get(node, 0)
        else:
            self.logger.warning("图为空，跳过节点中心性计算")
        
        return {
            'global_features': network_features,
            'user_network_features': user_network_features
        }
    
    def extract_temporal_features(self, data):
        """
        提取时间特征
        
        参数:
            data: 元数据
            
        返回:
            时间特征
        """
        self.logger.info("提取时间特征")
        
        # 用户活动时间分布 - 避免使用嵌套的defaultdict(lambda)
        user_activity = {}
        
        # 会话活动时间分布 - 避免使用嵌套的defaultdict(lambda)
        session_activity = {}
        
        # 处理每个会话
        for post_id, post_data in tqdm(data.items(), desc="分析时间模式"):
            comments = post_data.get('comments', [])
            
            # 初始化会话活动记录
            if post_id not in session_activity:
                session_activity[post_id] = {
                    'hour': {},
                    'day_of_week': {},
                    'month': {},
                    'time_of_day': {}
                }
            
            for comment in comments:
                user_id = comment.get('userId', '')
                created_time = comment.get('created', '')
                
                if not user_id or not created_time:
                    continue
                
                # 解析时间
                try:
                    dt = None
                    # 尝试不同的时间格式
                    for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
                        try:
                            dt = pd.to_datetime(created_time, format=fmt)
                            break
                        except ValueError:
                            continue
                    
                    if dt is None:
                        continue
                    
                    # 提取时间特征
                    hour = dt.hour
                    day_of_week = dt.weekday()  # 0-6，0表示星期一
                    month = dt.month
                    
                    # 定义时间段
                    if 5 <= hour < 12:
                        time_of_day = 'morning'
                    elif 12 <= hour < 17:
                        time_of_day = 'afternoon'
                    elif 17 <= hour < 22:
                        time_of_day = 'evening'
                    else:
                        time_of_day = 'night'
                    
                    # 更新用户活动时间分布
                    if user_id not in user_activity:
                        user_activity[user_id] = {
                            'hour': {},
                            'day_of_week': {},
                            'month': {},
                            'time_of_day': {}
                        }
                    
                    # 更新小时分布
                    if hour not in user_activity[user_id]['hour']:
                        user_activity[user_id]['hour'][hour] = 0
                    user_activity[user_id]['hour'][hour] += 1
                    
                    # 更新星期几分布
                    if day_of_week not in user_activity[user_id]['day_of_week']:
                        user_activity[user_id]['day_of_week'][day_of_week] = 0
                    user_activity[user_id]['day_of_week'][day_of_week] += 1
                    
                    # 更新月份分布
                    if month not in user_activity[user_id]['month']:
                        user_activity[user_id]['month'][month] = 0
                    user_activity[user_id]['month'][month] += 1
                    
                    # 更新时间段分布
                    if time_of_day not in user_activity[user_id]['time_of_day']:
                        user_activity[user_id]['time_of_day'][time_of_day] = 0
                    user_activity[user_id]['time_of_day'][time_of_day] += 1
                    
                    # 更新会话活动时间分布
                    # 更新小时分布
                    if hour not in session_activity[post_id]['hour']:
                        session_activity[post_id]['hour'][hour] = 0
                    session_activity[post_id]['hour'][hour] += 1
                    
                    # 更新星期几分布
                    if day_of_week not in session_activity[post_id]['day_of_week']:
                        session_activity[post_id]['day_of_week'][day_of_week] = 0
                    session_activity[post_id]['day_of_week'][day_of_week] += 1
                    
                    # 更新月份分布
                    if month not in session_activity[post_id]['month']:
                        session_activity[post_id]['month'][month] = 0
                    session_activity[post_id]['month'][month] += 1
                    
                    # 更新时间段分布
                    if time_of_day not in session_activity[post_id]['time_of_day']:
                        session_activity[post_id]['time_of_day'][time_of_day] = 0
                    session_activity[post_id]['time_of_day'][time_of_day] += 1
                    
                except Exception as e:
                    continue
        
        # 计算用户活动峰值时间
        user_peak_times = {}
        for user_id, activity in user_activity.items():
            # 找出活动最多的时间段
            peak_hour = None
            peak_hour_count = -1
            for hour, count in activity['hour'].items():
                if count > peak_hour_count:
                    peak_hour = hour
                    peak_hour_count = count
            
            peak_day = None
            peak_day_count = -1
            for day, count in activity['day_of_week'].items():
                if count > peak_day_count:
                    peak_day = day
                    peak_day_count = count
            
            peak_tod = None
            peak_tod_count = -1
            for tod, count in activity['time_of_day'].items():
                if count > peak_tod_count:
                    peak_tod = tod
                    peak_tod_count = count
            
            user_peak_times[user_id] = {
                'peak_hour': peak_hour,
                'peak_day': peak_day,
                'peak_time_of_day': peak_tod
            }
        
        # 计算会话活动峰值时间
        session_peak_times = {}
        for post_id, activity in session_activity.items():
            # 找出活动最多的时间段
            peak_hour = None
            peak_hour_count = -1
            for hour, count in activity['hour'].items():
                if count > peak_hour_count:
                    peak_hour = hour
                    peak_hour_count = count
            
            peak_day = None
            peak_day_count = -1
            for day, count in activity['day_of_week'].items():
                if count > peak_day_count:
                    peak_day = day
                    peak_day_count = count
            
            peak_tod = None
            peak_tod_count = -1
            for tod, count in activity['time_of_day'].items():
                if count > peak_tod_count:
                    peak_tod = tod
                    peak_tod_count = count
            
            session_peak_times[post_id] = {
                'peak_hour': peak_hour,
                'peak_day': peak_day,
                'peak_time_of_day': peak_tod
            }
        
        return {
            'user_activity': user_activity,
            'user_peak_times': user_peak_times,
            'session_activity': session_activity,
            'session_peak_times': session_peak_times
        }
    
    def extract_user_profile_features(self, data):
        """
        提取用户资料特征
        
        参数:
            data: 元数据
            
        返回:
            用户资料特征
        """
        self.logger.info("提取用户资料特征")
        
        user_profiles = {}
        
        # 提取用户资料信息
        for post_id, post_data in tqdm(data.items(), desc="提取用户资料"):
            comments = post_data.get('comments', [])
            is_bullying = post_data.get('label', 0)
            
            for comment in comments:
                user_id = comment.get('userId', '')
                
                if not user_id:
                    continue
                
                # 如果用户不在字典中，添加
                if user_id not in user_profiles:
                    user_profiles[user_id] = {
                        'username': comment.get('username', ''),
                        'comment_count': 0,
                        'bullying_session_count': 0,
                        'normal_session_count': 0,
                        'verified': comment.get('verified', '0'),
                        'description': comment.get('description', ''),
                        'private': comment.get('private', '0'),
                        'location': comment.get('location', ''),
                        'sessions': set()
                    }
                
                # 更新用户数据
                user_profiles[user_id]['comment_count'] += 1
                user_profiles[user_id]['sessions'].add(post_id)
                
                if is_bullying:
                    user_profiles[user_id]['bullying_session_count'] += 1
                else:
                    user_profiles[user_id]['normal_session_count'] += 1
        
        # 计算用户参与的会话总数
        for user_id, profile in user_profiles.items():
            profile['session_count'] = len(profile['sessions'])
            # 将集合转换为列表以便序列化
            profile['sessions'] = list(profile['sessions'])
            
            # 计算霸凌参与率
            total_sessions = profile['bullying_session_count'] + profile['normal_session_count']
            profile['bullying_ratio'] = profile['bullying_session_count'] / total_sessions if total_sessions > 0 else 0
        
        self.logger.info(f"提取了 {len(user_profiles)} 位用户的资料特征")
        
        return user_profiles
    
    def extract_features(self, data):
        """
        提取用户特征
        
        参数:
            data: 元数据
            
        返回:
            提取的特征
        """
        self.logger.info("开始提取用户特征")
        
        # 提取用户资料特征
        user_profiles = self.extract_user_profile_features(data)
        
        # 构建用户交互网络
        user_network, session_users = self.build_user_network(data)
        
        # 提取网络特征
        network_features = self.extract_network_features(user_network)
        
        # 提取时间特征
        temporal_features = self.extract_temporal_features(data)
        
        # 整合所有用户特征
        user_features = {}
        for user_id in user_profiles:
            # 基本资料特征
            user_features[user_id] = user_profiles[user_id].copy()
            
            # 添加网络特征
            if user_id in network_features['user_network_features']:
                user_features[user_id]['network_features'] = network_features['user_network_features'][user_id]
            
            # 添加时间特征
            if user_id in temporal_features['user_peak_times']:
                user_features[user_id]['temporal_features'] = temporal_features['user_peak_times'][user_id]
        
        # 计算会话级用户特征
        session_features = {}
        for post_id in data:
            # 获取会话中的用户
            users = list(session_users.get(post_id, set()))
            
            if not users:
                continue
            
            # 收集会话中用户的特征
            users_network_features = []
            users_bullying_ratio = []
            
            for user_id in users:
                if user_id in user_features and 'network_features' in user_features[user_id]:
                    users_network_features.append(user_features[user_id]['network_features'])
                if user_id in user_features:
                    users_bullying_ratio.append(user_features[user_id].get('bullying_ratio', 0))
            
            # 计算会话级特征
            session_feature = {
                'post_id': post_id,
                'user_count': len(users),
                'avg_user_degree': np.mean([f.get('total_degree', 0) for f in users_network_features]) if users_network_features else 0,
                'max_user_degree': np.max([f.get('total_degree', 0) for f in users_network_features]) if users_network_features else 0,
                'avg_user_pagerank': np.mean([f.get('pagerank', 0) for f in users_network_features]) if users_network_features else 0,
                'max_user_pagerank': np.max([f.get('pagerank', 0) for f in users_network_features]) if users_network_features else 0,
                'avg_bullying_ratio': np.mean(users_bullying_ratio) if users_bullying_ratio else 0,
                'max_bullying_ratio': np.max(users_bullying_ratio) if users_bullying_ratio else 0,
            }
            
            # 添加时间特征
            if post_id in temporal_features['session_peak_times']:
                session_feature['temporal_features'] = temporal_features['session_peak_times'][post_id]
            
            session_features[post_id] = session_feature
        
        # 存储特征
        self.user_features = user_features
        self.social_network_features = network_features
        self.temporal_features = temporal_features
        
        return {
            'user_features': user_features,
            'session_user_features': session_features,
            'network_features': network_features,
            'temporal_features': temporal_features
        }
    
    def get_feature_names(self):
        """
        获取特征名称列表
        
        返回:
            特征名称列表
        """
        feature_names = [
            # 会话用户特征
            'user_count',
            'avg_user_degree',
            'max_user_degree',
            'avg_user_pagerank',
            'max_user_pagerank',
            'avg_bullying_ratio',
            'max_bullying_ratio',
            
            # 时间特征
            'peak_hour',
            'peak_day',
            'peak_time_of_day',
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
        features_file = os.path.join(self.features_path, 'user_features.pkl')
        
        # 如果不强制重新处理且特征文件存在，则加载
        if not force_reprocess and os.path.exists(features_file):
            self.logger.info(f"从 {features_file} 加载已处理的用户特征")
            features = self.load_features(features_file)
            
            # 更新实例变量
            self.user_features = features.get('user_features', {})
            self.social_network_features = features.get('network_features', {})
            self.temporal_features = features.get('temporal_features', {})
            
            return features
        
        # 如果需要重新处理但未提供数据，则加载数据
        if data is None:
            metadata_file = os.path.join(self.config["metadata_data_path"], 'processed_metadata.json')
            self.logger.info(f"从 {metadata_file} 加载元数据")
            data = self.load_processed_data(metadata_file)
        
        # 提取特征
        features = self.extract_features(data)
        
        # 保存特征
        self.save_features(features, features_file)
        
        return features

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="用户特征提取")
    parser.add_argument("--force", action="store_true", help="强制重新处理")
    args = parser.parse_args()
    
    extractor = UserFeatureExtractor()
    features = extractor.get_processed_features(force_reprocess=args.force)
    
    print(f"提取了 {len(features['user_features'])} 位用户的特征")
    print(f"提取了 {len(features['session_user_features'])} 个会话的用户特征")

if __name__ == "__main__":
    main() 