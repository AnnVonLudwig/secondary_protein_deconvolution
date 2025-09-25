"""
数据预处理器
============

提供数据预处理功能。
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Optional


def normalize_spectrum(spectrum: np.ndarray, method: str = 'max') -> np.ndarray:
    """
    光谱归一化
    
    Parameters:
    -----------
    spectrum : np.ndarray
        输入光谱
    method : str
        归一化方法 ('max', 'area', 'minmax')
        
    Returns:
    --------
    normalized_spectrum : np.ndarray
        归一化后的光谱
    """
    if method == 'max':
        return spectrum / np.max(spectrum)
    elif method == 'area':
        return spectrum / np.trapz(spectrum)
    elif method == 'minmax':
        return (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def smooth_spectrum(spectrum: np.ndarray, window_length: int = 9, polyorder: int = 3) -> np.ndarray:
    """
    光谱平滑
    
    Parameters:
    -----------
    spectrum : np.ndarray
        输入光谱
    window_length : int
        平滑窗口长度
    polyorder : int
        多项式阶数
        
    Returns:
    --------
    smoothed_spectrum : np.ndarray
        平滑后的光谱
    """
    return savgol_filter(spectrum, window_length, polyorder)


def baseline_correction(spectrum: np.ndarray, method: str = 'polynomial', degree: int = 2) -> np.ndarray:
    """
    基线校正
    
    Parameters:
    -----------
    spectrum : np.ndarray
        输入光谱
    method : str
        校正方法 ('polynomial', 'rolling')
    degree : int
        多项式阶数
        
    Returns:
    --------
    corrected_spectrum : np.ndarray
        基线校正后的光谱
    """
    if method == 'polynomial':
        x = np.arange(len(spectrum))
        coeffs = np.polyfit(x, spectrum, degree)
        baseline = np.polyval(coeffs, x)
        return spectrum - baseline
    elif method == 'rolling':
        # 简单的滚动最小值基线校正
        window_size = len(spectrum) // 10
        baseline = np.minimum.accumulate(spectrum)
        return spectrum - baseline
    else:
        raise ValueError(f"Unknown baseline correction method: {method}")


def preprocess_spectrum(spectrum: np.ndarray, 
                       wavenumber_axis: Optional[np.ndarray] = None,
                       normalize: bool = True,
                       smooth: bool = True,
                       baseline_correct: bool = True) -> np.ndarray:
    """
    完整的光谱预处理流程
    
    Parameters:
    -----------
    spectrum : np.ndarray
        输入光谱
    wavenumber_axis : np.ndarray, optional
        波数轴
    normalize : bool
        是否归一化
    smooth : bool
        是否平滑
    baseline_correct : bool
        是否基线校正
        
    Returns:
    --------
    processed_spectrum : np.ndarray
        预处理后的光谱
    """
    processed = spectrum.copy()
    
    if baseline_correct:
        processed = baseline_correction(processed)
    
    if smooth:
        processed = smooth_spectrum(processed)
    
    if normalize:
        processed = normalize_spectrum(processed)
    
    return processed
