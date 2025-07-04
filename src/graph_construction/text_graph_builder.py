"""
文本图构建器

负责构建基于文本内容的图结构，包括用户节点、评论节点和词汇节点。
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter
import re
from .base_graph_builder import BaseGraphBuilder

class TextGraphBuilder(BaseGraphBuilder):
    """文本图构建器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化文本图构建器
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 文本处理配置
        self.min_word_freq = config.get('min_word_freq', 2)
        self.max_vocab_size = config.get('max_vocab_size', 5000)
        self.include_sentiment_edges = config.get('include_sentiment_edges', True)
        self.include_attack_edges = config.get('include_attack_edges', True)
        
        # 词汇表
        self.vocabulary = {}
        self.word_to_idx = {}
        
        # 攻击性词汇列表（简化版）
        self.attack_words = {
            'hate', 'stupid', 'idiot', 'loser', 'ugly', 'fat', 'dumb', 
            'kill', 'die', 'shut up', 'go away', 'pathetic', 'worthless',
            'freak', 'weirdo', 'creep', 'disgusting', 'gross'
        }
    
    def build_graph(self, data):
        """
        构建文本图
        基于真实数据格式重新实现
        """
        print("开始构建文本图...")
        
        # 从多模态对齐数据中提取信息
        aligned_data = data.get('aligned_data', {})
        if not aligned_data:
            print("❌ 未找到对齐数据")
            return None
        
        print(f"处理 {len(aligned_data)} 个帖子的数据")
        
        # 初始化节点和边存储
        self.user_nodes = {}
        self.comment_nodes = {}  
        self.word_nodes = {}
        self.video_nodes = {}  # 添加视频节点
        
        self.edges = {
            'user_posts_comment': [],
            'comment_contains_word': [],
            'user_interacts_user': [],
            'media_session_contains_comment': [],  # 媒体会话包含评论的边
            'video_contains_comment': []  # 视频包含评论的边（向后兼容）
        }
        
        # 初始化映射
        self.session_mappings = {
            'video_to_session': {},
            'session_to_videos': {},
            'comment_id_mapping': {},
            'original_to_internal': {},
            'internal_to_original': {}
        }
        
        # 第一步：收集所有用户和评论
        all_users = set()
        all_comments = []
        word_counter = {}
        
        for post_id, post_data in aligned_data.items():
            # 创建视频节点
            self._create_video_node(post_id, post_data)
            
            # 处理评论数据 - 修复数据结构匹配
            # 实际数据结构是 post_data['comments']，不是 post_data['text_data']['comments']
            comments = post_data.get('comments', [])
            
            for comment in comments:
                all_comments.append(comment)
                
                # 提取用户名（清理HTML标签）
                username = comment.get('username', 'unknown')
                if username:
                    import re
                    clean_username = re.sub(r'<[^>]+>', '', username).strip()
                    all_users.add(clean_username)
                
                # 提取词汇
                comment_text = comment.get('commentText', '')
                if comment_text:
                    words = self._extract_words(comment_text)
                    for word in words:
                        word_counter[word] = word_counter.get(word, 0) + 1
        
        print(f"提取到: {len(all_users)} 个用户, {len(all_comments)} 条评论")
        
        # 第二步：创建节点
        self._create_user_nodes(all_users, aligned_data)
        self._create_comment_nodes(all_comments, aligned_data)
        self._create_word_nodes(word_counter)
        
        # 第三步：创建边
        self._create_edges(aligned_data)
        
        # 第四步：构建PyTorch Geometric图
        graph = self._build_pytorch_graph()
        
        # 第五步：设置节点映射（用于异构图构建器）
        self.node_mappings = {
            'user': {username: node['id'] for username, node in self.user_nodes.items()},
            'comment': {comment_id: node['id'] for comment_id, node in self.comment_nodes.items()},
            'word': {word: node['id'] for word, node in self.word_nodes.items()},
            'video': {post_id: node['id'] for post_id, node in self.video_nodes.items()}
        }
        
        # 添加media_session映射
        if hasattr(self, 'media_session_nodes'):
            self.node_mappings['media_session'] = {post_id: node['id'] for post_id, node in self.media_session_nodes.items()}
        
        print(f"✅ 文本图构建完成:")
        print(f"   节点: user={len(self.user_nodes)}, comment={len(self.comment_nodes)}, word={len(self.word_nodes)}, video={len(self.video_nodes)}")
        if hasattr(self, 'media_session_nodes'):
            print(f"   媒体会话: {len(self.media_session_nodes)}")
        
        return graph

    def _create_video_node(self, post_id, post_data):
        """创建媒体会话节点（原video节点重命名为media_session）"""
        # 安全获取字符串属性，处理NaN和None
        def safe_str(value, default=''):
            if value is None or (hasattr(value, '__iter__') and value != value):  # NaN检查
                return default
            return str(value) if value is not None else default
        
        # 创建media_session节点而不是video节点
        if not hasattr(self, 'media_session_nodes'):
            self.media_session_nodes = {}
        
        self.media_session_nodes[post_id] = {
            'id': len(self.media_session_nodes),
            'post_id': post_id,
            'label': post_data.get('label', 0),  # 霸凌标签
            'caption': safe_str(post_data.get('caption')),
            'username': safe_str(post_data.get('username')),
            'video_url': safe_str(post_data.get('video_url')),
            'type': 'media_session'
        }
        
        # 保持video_nodes的兼容性（指向media_session_nodes）
        self.video_nodes[post_id] = self.media_session_nodes[post_id]
        
        # 添加到会话映射
        session_id = f"media_session_{post_id}"
        self.session_mappings['video_to_session'][post_id] = session_id
        self.session_mappings['session_to_videos'][session_id] = post_id

    def _create_user_nodes(self, all_users, aligned_data):
        """基于真实数据创建用户节点"""
        
        # 统计每个用户的活动
        user_stats = {}
        for post_id, post_data in aligned_data.items():
            comments = post_data.get('comments', [])
            for comment in comments:
                username = comment.get('username', 'unknown')
                if username:
                    import re
                    clean_username = re.sub(r'<[^>]+>', '', username).strip()
                    
                    if clean_username not in user_stats:
                        user_stats[clean_username] = {
                            'comment_count': 0,
                            'first_seen': comment.get('created'),
                            'last_seen': comment.get('created'),
                            'verified': comment.get('verified', '0') == '1',
                            'posts': set()
                        }
                    
                    user_stats[clean_username]['comment_count'] += 1
                    user_stats[clean_username]['posts'].add(post_id)
                    
                    # 更新时间范围
                    created = comment.get('created')
                    if created:
                        if not user_stats[clean_username]['first_seen'] or created < user_stats[clean_username]['first_seen']:
                            user_stats[clean_username]['first_seen'] = created
                        if not user_stats[clean_username]['last_seen'] or created > user_stats[clean_username]['last_seen']:
                            user_stats[clean_username]['last_seen'] = created
        
        # 创建用户节点
        for idx, username in enumerate(sorted(all_users)):
            stats = user_stats.get(username, {})
            self.user_nodes[username] = {
                'id': idx,
                'username': username,
                'comment_count': stats.get('comment_count', 0),
                'post_count': len(stats.get('posts', set())),
                'verified': stats.get('verified', False),
                'first_seen': stats.get('first_seen', ''),
                'last_seen': stats.get('last_seen', ''),
                'type': 'user'
            }
        
    def _create_comment_nodes(self, all_comments, aligned_data):
        """基于真实数据创建评论节点"""
        
        for idx, comment in enumerate(all_comments):
            comment_id = comment.get('commentId', f"comment_{idx}")
            
            # 清理用户名
            username = comment.get('username', 'unknown')
            import re
            clean_username = re.sub(r'<[^>]+>', '', username).strip()
            
            self.comment_nodes[comment_id] = {
                'id': idx,
                'comment_id': comment_id,
                'comment_text': comment.get('commentText', ''),
                'username': clean_username,
                'created': comment.get('created', ''),
                'post_id': comment.get('postId', ''),
                'user_id': comment.get('userId', ''),
                'verified': comment.get('verified', '0') == '1',
                'location': comment.get('location', ''),
                    'type': 'comment'
                }
            
            # 添加评论ID映射
            self.session_mappings['comment_id_mapping'][comment_id] = idx

    def _create_word_nodes(self, word_counter):
        """创建词汇节点"""
        
        # 选择前1000个高频词汇
        top_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)[:1000]
        
        for idx, (word, freq) in enumerate(top_words):
            self.word_nodes[word] = {
                'id': idx,
                'word': word,
                'frequency': freq,
                'is_attack': self._is_attack_word(word),
                'type': 'word'
            }
        
        print(f"创建词汇节点: {len(self.word_nodes)} 个高频词汇")

    def _extract_words(self, text):
        """从文本中提取词汇"""
        import re
        
        # 清理文本
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 分词
        words = text.split()
        
        # 过滤短词和停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = [word for word in words if len(word) >= 3 and word not in stop_words]
        
        return words

    def _is_attack_word(self, word):
        """判断是否为攻击性词汇"""
        attack_words = {
            'bitch', 'fuck', 'shit', 'damn', 'ass', 'nigga', 'niggas', 'dick', 
            'stupid', 'dumb', 'idiot', 'hate', 'kill', 'die', 'ugly', 'fat',
            'retard', 'gay', 'faggot', 'whore', 'slut', 'cunt'
        }
        return word.lower() in attack_words

    def _create_edges(self, aligned_data):
        """创建图的边"""
        
        # 1. 用户-评论边 (user_posts_comment)
        for post_id, post_data in aligned_data.items():
            comments = post_data.get('comments', [])
            for comment in comments:
                username = comment.get('username', 'unknown')
                import re
                clean_username = re.sub(r'<[^>]+>', '', username).strip()
                comment_id = comment.get('commentId')
                
                if clean_username in self.user_nodes and comment_id in self.comment_nodes:
                    user_idx = self.user_nodes[clean_username]['id']
                    comment_idx = self.comment_nodes[comment_id]['id']
                    self.edges['user_posts_comment'].append((user_idx, comment_idx))
        
        # 2. 评论-词汇边 (comment_contains_word)  
        for comment_id, comment_node in self.comment_nodes.items():
            comment_text = comment_node['comment_text']
            words = self._extract_words(comment_text)
            comment_idx = comment_node['id']
            
            for word in words:
                if word in self.word_nodes:
                    word_idx = self.word_nodes[word]['id']
                    self.edges['comment_contains_word'].append((comment_idx, word_idx))
        
        # 3. 媒体会话-评论边 (media_session_contains_comment)
        if hasattr(self, 'media_session_nodes'):
            for post_id, post_data in aligned_data.items():
                if post_id in self.media_session_nodes:
                    session_idx = self.media_session_nodes[post_id]['id']
                    comments = post_data.get('comments', [])
                    
                    for comment in comments:
                        comment_id = comment.get('commentId')
                        if comment_id in self.comment_nodes:
                            comment_idx = self.comment_nodes[comment_id]['id']
                            self.edges['media_session_contains_comment'].append((session_idx, comment_idx))
        
        # 为了向后兼容，保持video_contains_comment边
        for post_id, post_data in aligned_data.items():
            if post_id in self.video_nodes:
                video_idx = self.video_nodes[post_id]['id']
                comments = post_data.get('comments', [])
                
                for comment in comments:
                    comment_id = comment.get('commentId')
                    if comment_id in self.comment_nodes:
                        comment_idx = self.comment_nodes[comment_id]['id']
                        self.edges['video_contains_comment'].append((video_idx, comment_idx))
        
        # 4. 用户互动边 (基于共同参与的帖子)
        user_posts = {}
        for username, user_node in self.user_nodes.items():
            user_posts[username] = set()
        
        for post_id, post_data in aligned_data.items():
            comments = post_data.get('comments', [])
            post_users = set()
            for comment in comments:
                username = comment.get('username', 'unknown')
                import re
                clean_username = re.sub(r'<[^>]+>', '', username).strip()
                if clean_username in self.user_nodes:
                    post_users.add(clean_username)
                    user_posts[clean_username].add(post_id)
            
            # 为同一帖子的用户创建互动边
            post_users_list = list(post_users)
            for i in range(len(post_users_list)):
                for j in range(i+1, len(post_users_list)):
                    user1, user2 = post_users_list[i], post_users_list[j]
                    user1_idx = self.user_nodes[user1]['id']
                    user2_idx = self.user_nodes[user2]['id']
                    self.edges['user_interacts_user'].append((user1_idx, user2_idx))
    
        print(f"创建边: user_posts_comment={len(self.edges['user_posts_comment'])}, comment_contains_word={len(self.edges['comment_contains_word'])}, video_contains_comment={len(self.edges['video_contains_comment'])}, user_interacts_user={len(self.edges['user_interacts_user'])}")

    def _build_pytorch_graph(self):
        """构建PyTorch Geometric异构图"""
        import torch
        from torch_geometric.data import HeteroData
        
        graph = HeteroData()
        
        # 添加节点类型和特征
        
        # 用户节点
        user_features = []
        for username in sorted(self.user_nodes.keys()):
            user = self.user_nodes[username]
            features = [
                user['comment_count'],
                user['post_count'], 
                float(user['verified']),
                len(user['username']),
                1.0  # 其他特征占位
            ]
            user_features.append(features)
        
        if user_features:
            graph['user'].x = torch.tensor(user_features, dtype=torch.float)
        
        # 评论节点 - 匹配子图提取器期望的特征结构
        comment_features = []
        for comment_id in sorted(self.comment_nodes.keys(), key=lambda x: self.comment_nodes[x]['id']):
            comment = self.comment_nodes[comment_id]
            comment_text = comment['comment_text']
            
            # 计算高级特征
            exclamation_count = comment_text.count('!')
            question_count = comment_text.count('?')
            uppercase_chars = sum(1 for c in comment_text if c.isupper())
            total_chars = len(comment_text)
            uppercase_ratio = uppercase_chars / max(total_chars, 1)
            
            # 计算攻击词相关特征
            words = self._extract_words(comment_text)
            attack_words = [w for w in words if self._is_attack_word(w)]
            attack_word_count = len(attack_words)
            attack_word_ratio = attack_word_count / max(len(words), 1)
            
            # 特征向量：[文本长度, 词数, 感叹号数, 问号数, 大写比例, 攻击词数, 攻击词比例, 内部ID]
            features = [
                len(comment_text),                    # 0: 文本长度
                len(words),                          # 1: 词数
                exclamation_count,                   # 2: 感叹号数
                question_count,                      # 3: 问号数
                uppercase_ratio,                     # 4: 大写比例
                attack_word_count,                   # 5: 攻击词数
                attack_word_ratio,                   # 6: 攻击词比例
                float(comment['id'])                 # 7: 内部ID
            ]
            comment_features.append(features)
        
        if comment_features:
            graph['comment'].x = torch.tensor(comment_features, dtype=torch.float)
        
        # 词汇节点
        word_features = []
        for word in sorted(self.word_nodes.keys(), key=lambda x: self.word_nodes[x]['id']):
            word_node = self.word_nodes[word]
            features = [
                word_node['frequency'],
                float(word_node['is_attack']),
                len(word),
                1.0  # 其他特征占位
            ]
            word_features.append(features)
        
        if word_features:
            graph['word'].x = torch.tensor(word_features, dtype=torch.float)
        
        # 媒体会话节点 ⭐ 包含霸凌标签
        if hasattr(self, 'media_session_nodes') and self.media_session_nodes:
            media_session_features = []
            for post_id in sorted(self.media_session_nodes.keys(), key=lambda x: self.media_session_nodes[x]['id']):
                session = self.media_session_nodes[post_id]
                
                # 安全处理字符串长度
                def safe_len(value):
                    if value is None:
                        return 0
                    if isinstance(value, str):
                        return len(value)
                    return 0
                
                features = [
                    float(session['label']),  # 霸凌标签 ⭐ 第一个特征
                    safe_len(session['caption']),
                    safe_len(session['username']),
                    1.0 if session['video_url'] else 0.0,
                    1.0  # 其他特征占位
                ]
                media_session_features.append(features)
            
            graph['media_session'].x = torch.tensor(media_session_features, dtype=torch.float)
        
        # 为了向后兼容，也创建video节点（指向media_session）
        if hasattr(self, 'video_nodes') and self.video_nodes:
            video_features = []
            for post_id in sorted(self.video_nodes.keys(), key=lambda x: self.video_nodes[x]['id']):
                video = self.video_nodes[post_id]
                
                # 安全处理字符串长度
                def safe_len(value):
                    if value is None:
                        return 0
                    if isinstance(value, str):
                        return len(value)
                    return 0
                
                features = [
                    float(video['label']),  # 霸凌标签 ⭐ 第一个特征（已修复）
                    safe_len(video['caption']),
                    safe_len(video['username']),
                    1.0 if video['video_url'] else 0.0,
                    1.0  # 其他特征占位
                ]
                video_features.append(features)
            
            if video_features:
                graph['video'].x = torch.tensor(video_features, dtype=torch.float)
        
        # 添加边
        for edge_type, edge_list in self.edges.items():
            if edge_list:
                if edge_type == 'user_posts_comment':
                    graph['user', 'posts', 'comment'].edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                elif edge_type == 'comment_contains_word':
                    graph['comment', 'contains', 'word'].edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                elif edge_type == 'media_session_contains_comment':
                    graph['media_session', 'contains', 'comment'].edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                elif edge_type == 'video_contains_comment':
                    graph['video', 'contains', 'comment'].edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                elif edge_type == 'user_interacts_user':
                    graph['user', 'interacts', 'user'].edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        return graph 

    def get_session_mapping(self):
        """返回会话映射信息"""
        return self.session_mappings 