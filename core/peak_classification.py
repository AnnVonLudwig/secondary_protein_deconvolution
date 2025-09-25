"""
峰值分类模块
============

提供峰值分类和结构分析功能。
"""

import numpy as np
from typing import List, Tuple, Dict
from config.analysis_config import STRUCTURE_CLASSIFICATION_CONFIG


def classify_peak(peak_position: float, 
                 tolerance: float = None) -> str:
    """
    分类单个峰值
    
    Parameters:
    -----------
    peak_position : float
        峰值位置
    tolerance : float, optional
        分类容差
        
    Returns:
    --------
    peak_type : str
        峰值类型
    """
    if tolerance is None:
        tolerance = STRUCTURE_CLASSIFICATION_CONFIG['classification_tolerance']
    
    structure_peaks = STRUCTURE_CLASSIFICATION_CONFIG['structure_peaks']
    
    for peak_type, positions in structure_peaks.items():
        for pos in positions:
            if abs(peak_position - pos) <= tolerance:
                return peak_type
    
    return 'unknown'


def classify_peaks(peak_positions: np.ndarray, 
                  tolerance: float = None) -> Tuple[List[str], np.ndarray]:
    """
    批量分类峰值
    
    Parameters:
    -----------
    peak_positions : np.ndarray
        峰值位置数组
    tolerance : float, optional
        分类容差
        
    Returns:
    --------
    peak_types : list
        峰值类型列表
    peak_gammas : np.ndarray
        默认线宽数组
    """
    peak_types = []
    peak_gammas = []
    
    default_gamma = STRUCTURE_CLASSIFICATION_CONFIG['default_gamma']
    
    for pos in peak_positions:
        peak_type = classify_peak(pos, tolerance)
        peak_types.append(peak_type)
        peak_gammas.append(default_gamma.get(peak_type, 10.0))
    
    return peak_types, np.array(peak_gammas)


def calculate_structure_ratios(peak_types: List[str], 
                             amplitudes: np.ndarray) -> Dict[str, float]:
    """
    计算结构比例
    
    Parameters:
    -----------
    peak_types : list
        峰值类型列表
    amplitudes : np.ndarray
        峰值振幅数组
        
    Returns:
    --------
    ratios : dict
        结构比例字典
    """
    # 计算各结构类型的总强度
    beta_intensity = sum(amp for amp, ptype in zip(amplitudes, peak_types) 
                        if ptype in ['beta_sheet', 'beta_turn'])
    alpha_intensity = sum(amp for amp, ptype in zip(amplitudes, peak_types) 
                         if ptype == 'alpha_helix')
    unknown_intensity = sum(amp for amp, ptype in zip(amplitudes, peak_types) 
                           if ptype == 'unknown')
    
    total_intensity = sum(amplitudes)
    
    # 计算比例
    ratios = {
        'beta_alpha_ratio': beta_intensity / alpha_intensity if alpha_intensity > 0 else float('inf'),
        'beta_intensity': beta_intensity,
        'alpha_intensity': alpha_intensity,
        'unknown_intensity': unknown_intensity,
        'total_intensity': total_intensity,
        'beta_percentage': (beta_intensity / total_intensity * 100) if total_intensity > 0 else 0,
        'alpha_percentage': (alpha_intensity / total_intensity * 100) if total_intensity > 0 else 0,
        'unknown_percentage': (unknown_intensity / total_intensity * 100) if total_intensity > 0 else 0
    }
    
    return ratios


def get_peak_statistics(peak_positions: np.ndarray, 
                       peak_types: List[str], 
                       amplitudes: np.ndarray) -> Dict[str, Dict]:
    """
    获取峰值统计信息
    
    Parameters:
    -----------
    peak_positions : np.ndarray
        峰值位置数组
    peak_types : list
        峰值类型列表
    amplitudes : np.ndarray
        峰值振幅数组
        
    Returns:
    --------
    stats : dict
        统计信息字典
    """
    stats = {}
    
    for peak_type in set(peak_types):
        type_indices = [i for i, ptype in enumerate(peak_types) if ptype == peak_type]
        
        if type_indices:
            type_positions = peak_positions[type_indices]
            type_amplitudes = amplitudes[type_indices]
            
            stats[peak_type] = {
                'count': len(type_indices),
                'mean_position': np.mean(type_positions),
                'std_position': np.std(type_positions),
                'mean_amplitude': np.mean(type_amplitudes),
                'std_amplitude': np.std(type_amplitudes),
                'positions': type_positions.tolist(),
                'amplitudes': type_amplitudes.tolist()
            }
    
    return stats
