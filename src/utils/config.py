"""
Configuration file containing all required parameters for the project
"""

# Data paths
DATA_CONFIG = {
    "raw_data_path": "data/raw",
    "processed_data_path": "data/processed",
    "vine_labeled_data": "data/raw/vine_labeled_cyberbullying_data.csv",
    "vine_comments_data": "data/raw/sampled_post-comments_vine.json",
    "urls_to_postids": "data/raw/urls_to_postids.txt",
    
    # Processed data paths
    "text_data_path": "data/processed/text",
    "video_data_path": "data/processed/video", 
    "metadata_data_path": "data/processed/metadata",
    "aligned_data_path": "data/processed/multimodal",
    
    # Feature data paths
    "features_path": "data/features",
    
    # Graph data paths
    "graphs_path": "data/graphs",
    
    # Subgraph data paths
    "subgraphs_path": "data/subgraphs",
    
    # Prototype data paths
    "prototypes_path": "data/prototypes",
    
    # Model data paths
    "models_path": "data/models",
}

# Data processing parameters
PREPROCESSING_CONFIG = {
    "train_test_split_ratio": 0.7,  # Training set ratio
    "val_test_split_ratio": 0.67,   # Validation set ratio in the remaining data (equivalent to 20% of the total)
    "random_seed": 42,
    "max_comment_length": 100,
    "min_word_frequency": 5,
}

# Feature extraction parameters
FEATURE_CONFIG = {
    # Text features
    "bert_model": "bert-base-uncased",
    "bert_max_length": 128,
    "sentiment_analyzer": "vader",
    "use_pretrained_embeddings": True,
    
    # Video features  
    "video_feature_dim": 512,
    "extract_video_features": False,  # Set to False for now as we don't have actual videos
    
    # User features
    "user_feature_dim": 64,
    "social_network_features": True,
    
    # Multimodal features
    "final_feature_dim": 776,
    "feature_fusion_method": "concatenation",
    "normalize_features": True,
    "normalization_method": "minmax",
}

# Alignment configuration for multimodal processing
ALIGNMENT_CONFIG = {
    "alignment_method": "temporal",
    "time_window": 3600,  # 1 hour in seconds
    "max_alignment_distance": 7200,  # 2 hours in seconds
    "min_confidence": 0.5,
}

# Graph construction parameters
GRAPH_CONFIG = {
    "edge_types": ["C-U-C", "C-W-C", "C-T-C", "C-L-C", "C-E-C"],  # Comment-User-Comment, Comment-Word-Comment, Comment-Time-Comment, Comment-Location-Comment, Comment-Emotion-Comment
    "edge_types_detailed": [
        # Original relation types
        "user", "word", "time", "location",
        # Emotion relation types
        "sentiment", "emotion_anger", "emotion_fear", "emotion_joy", 
        "emotion_sadness", "emotion_disgust", "emotion_trust", 
        "emotion_surprise", "emotion_anticipation",
        "aggression", "bullying_language", "swear"
    ],
    "relation_thresholds": {
        "user": 0.6,      # C-U-C threshold recommended in the paper
        "word": 1.0,      # C-W-C threshold recommended in the paper
        "time": 0.3,      # C-T-C threshold recommended in the paper
        "location": 0.2,  # C-L-C threshold recommended in the paper
        # Default thresholds for emotion relations
        "sentiment": 0.5,
        "emotion_anger": 0.6,
        "emotion_fear": 0.4,
        "emotion_joy": 0.3,
        "emotion_sadness": 0.5,
        "emotion_disgust": 0.6,
        "emotion_trust": 0.3,
        "emotion_surprise": 0.3,
        "emotion_anticipation": 0.3,
        "aggression": 0.7,
        "bullying_language": 0.7,
        "swear": 0.6
    },
    "subgraph_size": 6,              # Subgraph size recommended in the paper
    "central_nodes_count": 10,       # Number of top central nodes to select
}

# Model parameters
MODEL_CONFIG = {
    "embedding_dim": 64,             # Embedding dimension recommended in the paper
    "hidden_dim": 128,
    "gat_heads": 3,                  # Number of attention heads recommended in the paper
    "dropout": 0.1,
    "learning_rate": 0.001,          # Learning rate recommended in the paper
    "weight_decay": 5e-4,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 5,    # Early stopping patience recommended in the paper
}

# Reinforcement learning parameters
RL_CONFIG = {
    "gamma": 0.99,                  # Discount factor
    "actor_lr": 1e-4,               # Actor learning rate
    "critic_lr": 1e-3,              # Critic learning rate
    "eps_clip": 0.2,                # PPO clip parameter
    "k_epochs": 4,                  # Number of epochs per update
}

# Visualization parameters
VISUALIZATION_CONFIG = {
    "figure_size": (10, 6),
    "dpi": 300,
    "font_size": 12,
    "line_width": 2,
} 