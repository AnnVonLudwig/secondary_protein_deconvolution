"""
数据配置
========

包含数据加载和预处理相关的配置参数。
"""

# 数据路径配置
DATA_PATHS = {
    'raw_data': 'already_processed_real_data/',
    'processed_data': 'data/processed/',
    'output': 'output/',
    'models': 'models/saved/'
}

# 数据预处理参数
PREPROCESSING_CONFIG = {
    'normalize': True,
    'smooth': True,
    'baseline_correction': True,
    'noise_reduction': True
}

# 数据生成参数
DATA_GENERATION_CONFIG = {
    'wavenumber_range': (1570, 1730),
    'wavenumber_step': 0.1,
    'n_samples': 10000,
    'noise_level': 0.05
}
