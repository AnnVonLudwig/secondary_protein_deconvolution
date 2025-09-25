"""
峰值检测模块
============

提供各种峰值检测算法。
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Tuple, List, Optional, Dict


def detect_peaks_basic(y: np.ndarray, 
                      x_axis: np.ndarray,
                      threshold: float = 0.005,
                      min_distance: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    基础峰值检测
    
    Parameters:
    -----------
    y : np.ndarray
        光谱数据
    x_axis : np.ndarray
        波数轴
    threshold : float
        检测阈值
    min_distance : int
        最小峰值距离
        
    Returns:
    --------
    peak_indices : np.ndarray
        峰值索引
    y_smooth : np.ndarray
        平滑后的光谱
    """
    # 平滑处理
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    # 检测峰值
    peaks, _ = find_peaks(y_smooth, 
                         height=threshold * np.max(y_smooth),
                         distance=min_distance)
    
    return peaks, y_smooth


def detect_peaks_derivative(y: np.ndarray, 
                           x_axis: np.ndarray,
                           threshold: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于导数的峰值检测
    
    Parameters:
    -----------
    y : np.ndarray
        光谱数据
    x_axis : np.ndarray
        波数轴
    threshold : float
        检测阈值
        
    Returns:
    --------
    peak_indices : np.ndarray
        峰值索引
    y_smooth : np.ndarray
        平滑后的光谱
    """
    # 平滑处理
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    # 计算一阶和二阶导数
    dy = np.gradient(y_smooth)
    d2y = np.gradient(dy)
    
    # 寻找峰值（一阶导数为0，二阶导数为负）
    candidates = []
    for i in range(1, len(y_smooth)-1):
        if (dy[i-1] > 0 and dy[i+1] < 0 and 
            d2y[i] < 0 and 
            y_smooth[i] > threshold * np.max(y_smooth)):
            candidates.append(i)
    
    return np.array(candidates), y_smooth


def detect_peaks_adaptive(y: np.ndarray, 
                         x_axis: np.ndarray,
                         window_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    自适应峰值检测
    
    Parameters:
    -----------
    y : np.ndarray
        光谱数据
    x_axis : np.ndarray
        波数轴
    window_size : int
        自适应窗口大小
        
    Returns:
    --------
    peak_indices : np.ndarray
        峰值索引
    y_smooth : np.ndarray
        平滑后的光谱
    """
    # 平滑处理
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    # 自适应阈值
    adaptive_threshold = np.zeros_like(y_smooth)
    for i in range(len(y_smooth)):
        start = max(0, i - window_size // 2)
        end = min(len(y_smooth), i + window_size // 2)
        local_max = np.max(y_smooth[start:end])
        adaptive_threshold[i] = 0.1 * local_max
    
    # 检测峰值
    peaks, _ = find_peaks(y_smooth, height=adaptive_threshold)
    
    return peaks, y_smooth


def detect_all_peaks(y: np.ndarray, 
                    x_axis: np.ndarray,
                    methods: List[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    使用多种方法检测峰值
    
    Parameters:
    -----------
    y : np.ndarray
        光谱数据
    x_axis : np.ndarray
        波数轴
    methods : list, optional
        检测方法列表
        
    Returns:
    --------
    results : dict
        各方法的检测结果
    """
    if methods is None:
        methods = ['basic', 'derivative', 'adaptive']
    
    results = {}
    
    if 'basic' in methods:
        results['basic'] = detect_peaks_basic(y, x_axis)
    
    if 'derivative' in methods:
        results['derivative'] = detect_peaks_derivative(y, x_axis)
    
    if 'adaptive' in methods:
        results['adaptive'] = detect_peaks_adaptive(y, x_axis)
    
    return results