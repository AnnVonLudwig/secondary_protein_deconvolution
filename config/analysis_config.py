"""
分析配置
========

包含光谱分析相关的配置参数。
"""

# 峰值检测参数
PEAK_DETECTION_CONFIG = {
    'threshold': 0.005,
    'window': 1,
    'min_distance': 1,
    'savgol_window': 9,
    'savgol_polyorder': 3
}

# 结构分类参数
STRUCTURE_CLASSIFICATION_CONFIG = {
    'structure_peaks': {
        'alpha_helix': [1656],
        'beta_sheet': [1624, 1627, 1633, 1638, 1642, 1691, 1696],
        'beta_turn': [1667, 1675, 1680, 1685]
    },
    'classification_tolerance': 2.0,
    'default_gamma': {
        'alpha_helix': 8.0,
        'beta_sheet': 10.0,
        'beta_turn': 12.0,
        'unknown': 10.0
    }
}

# 积分分析参数
INTEGRAL_ANALYSIS_CONFIG = {
    'peak_regions': {
        'alpha_helix': [(1654, 1658)],
        'beta_sheet': [(1622, 1644), (1689, 1698)],
        'beta_turn': [(1665, 1687)]
    },
    'integration_method': 'trapz'
}

# NNLS分析参数
NNLS_CONFIG = {
    'max_iterations': 1000,
    'tolerance': 1e-6,
    'apply_constraints': True,
    'min_amplitude': 0.0,
    'max_amplitude': 10.0
}
