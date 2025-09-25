"""
模型配置
========

包含深度学习模型相关的配置参数。
"""

# 模型训练参数
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 10,
    'validation_split': 0.2
}

# 模型架构参数
MODEL_CONFIG = {
    'input_length': 1601,
    'hidden_dim': 512,
    'num_heads': 4,
    'num_layers': 3,
    'dropout': 0.1
}

# 数据增强参数
AUGMENTATION_CONFIG = {
    'noise_std': 0.05,
    'baseline_drift': 0.02,
    'peak_shift_std': 0.5
}
