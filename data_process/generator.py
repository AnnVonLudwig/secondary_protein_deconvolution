"""
数据生成器
==========

提供合成数据生成功能。
"""

import numpy as np
from typing import Tuple, Dict, List


def generate_synthetic_spectrum(wavenumber_axis: np.ndarray, 
                              n_peaks: int = 10,
                              noise_level: float = 0.05) -> np.ndarray:
    """
    生成合成光谱数据
    
    Parameters:
    -----------
    wavenumber_axis : np.ndarray
        波数轴
    n_peaks : int
        峰值数量
    noise_level : float
        噪声水平
        
    Returns:
    --------
    spectrum : np.ndarray
        合成光谱
    """
    spectrum = np.zeros_like(wavenumber_axis)
    
    # 生成随机峰值
    for _ in range(n_peaks):
        center = np.random.uniform(wavenumber_axis.min(), wavenumber_axis.max())
        amplitude = np.random.uniform(0.5, 2.0)
        gamma = np.random.uniform(5, 15)
        
        # 洛伦兹峰
        peak = amplitude / (1 + ((wavenumber_axis - center) / (gamma/2))**2)
        spectrum += peak
    
    # 添加噪声
    noise = np.random.normal(0, noise_level, len(spectrum))
    spectrum += noise
    
    return spectrum


def generate_protein_spectrum(wavenumber_axis: np.ndarray,
                            alpha_content: float = 0.3,
                            beta_content: float = 0.4,
                            turn_content: float = 0.3) -> Dict[str, np.ndarray]:
    """
    生成蛋白质光谱数据
    
    Parameters:
    -----------
    wavenumber_axis : np.ndarray
        波数轴
    alpha_content : float
        α-螺旋含量
    beta_content : float
        β-折叠含量
    turn_content : float
        β-转角含量
        
    Returns:
    --------
    spectra_dict : dict
        包含各组分光谱的字典
    """
    # 结构峰值位置
    alpha_centers = [1656]
    beta_centers = [1624, 1627, 1633, 1638, 1642, 1691, 1696]
    turn_centers = [1667, 1675, 1680, 1685]
    
    # 生成各组分光谱
    alpha_spectrum = np.zeros_like(wavenumber_axis)
    beta_spectrum = np.zeros_like(wavenumber_axis)
    turn_spectrum = np.zeros_like(wavenumber_axis)
    
    # α-螺旋
    for center in alpha_centers:
        amplitude = alpha_content * np.random.uniform(0.8, 1.2)
        gamma = np.random.uniform(6, 12)
        peak = amplitude / (1 + ((wavenumber_axis - center) / (gamma/2))**2)
        alpha_spectrum += peak
    
    # β-折叠
    for center in beta_centers:
        if np.random.random() > 0.3:  # 70%概率出现
            amplitude = beta_content * np.random.uniform(0.5, 1.5)
            gamma = np.random.uniform(8, 15)
            peak = amplitude / (1 + ((wavenumber_axis - center) / (gamma/2))**2)
            beta_spectrum += peak
    
    # β-转角
    for center in turn_centers:
        if np.random.random() > 0.4:  # 60%概率出现
            amplitude = turn_content * np.random.uniform(0.3, 1.0)
            gamma = np.random.uniform(10, 18)
            peak = amplitude / (1 + ((wavenumber_axis - center) / (gamma/2))**2)
            turn_spectrum += peak
    
    # 合成总光谱
    composite = alpha_spectrum + beta_spectrum + turn_spectrum
    
    return {
        'alpha_helix': alpha_spectrum,
        'beta_sheet': beta_spectrum,
        'beta_turn': turn_spectrum,
        'composite': composite,
        'wavenumber_axis': wavenumber_axis
    }
