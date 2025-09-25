"""
光谱函数模块
============

提供光谱分析相关的数学函数。
"""

import numpy as np
from typing import Union, List, Tuple


def lorentzian(x: np.ndarray, x0: float, gamma: float) -> np.ndarray:
    """
    洛伦兹函数
    
    Parameters:
    -----------
    x : np.ndarray
        波数轴
    x0 : float
        峰值中心
    gamma : float
        线宽参数 (FWHM)
        
    Returns:
    --------
    y : np.ndarray
        洛伦兹函数值
    """
    return 1 / (1 + ((x - x0) / (gamma / 2))**2)


def gaussian(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    """
    高斯函数
    
    Parameters:
    -----------
    x : np.ndarray
        波数轴
    x0 : float
        峰值中心
    sigma : float
        标准差
        
    Returns:
    --------
    y : np.ndarray
        高斯函数值
    """
    return np.exp(-((x - x0) / sigma)**2 / 2)


def voigt(x: np.ndarray, x0: float, gamma: float, sigma: float) -> np.ndarray:
    """
    Voigt函数 (高斯和洛伦兹的卷积)
    
    Parameters:
    -----------
    x : np.ndarray
        波数轴
    x0 : float
        峰值中心
    gamma : float
        洛伦兹线宽
    sigma : float
        高斯标准差
        
    Returns:
    --------
    y : np.ndarray
        Voigt函数值
    """
    # 简化的Voigt函数实现
    lorentz = lorentzian(x, x0, gamma)
    gauss = gaussian(x, x0, sigma)
    return (lorentz + gauss) / 2


def lorentzian_area(amplitudes: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """
    计算洛伦兹峰面积
    
    Parameters:
    -----------
    amplitudes : np.ndarray
        峰值振幅
    gammas : np.ndarray
        线宽参数
        
    Returns:
    --------
    areas : np.ndarray
        峰值面积
    """
    return np.pi * amplitudes * gammas / 2


def gaussian_area(amplitudes: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """
    计算高斯峰面积
    
    Parameters:
    -----------
    amplitudes : np.ndarray
        峰值振幅
    sigmas : np.ndarray
        标准差
        
    Returns:
    --------
    areas : np.ndarray
        峰值面积
    """
    return amplitudes * sigmas * np.sqrt(2 * np.pi)


def build_basis_functions(wavenumber_axis: np.ndarray, 
                         peak_centers: List[float],
                         peak_gammas: List[float],
                         function_type: str = 'lorentzian') -> np.ndarray:
    """
    构建基函数矩阵
    
    Parameters:
    -----------
    wavenumber_axis : np.ndarray
        波数轴
    peak_centers : list
        峰值中心列表
    peak_gammas : list
        线宽参数列表
    function_type : str
        函数类型 ('lorentzian', 'gaussian', 'voigt')
        
    Returns:
    --------
    basis_matrix : np.ndarray
        基函数矩阵
    """
    n_points = len(wavenumber_axis)
    n_peaks = len(peak_centers)
    
    basis_matrix = np.zeros((n_points, n_peaks))
    
    for i, (center, gamma) in enumerate(zip(peak_centers, peak_gammas)):
        if function_type == 'lorentzian':
            basis_matrix[:, i] = lorentzian(wavenumber_axis, center, gamma)
        elif function_type == 'gaussian':
            basis_matrix[:, i] = gaussian(wavenumber_axis, center, gamma)
        elif function_type == 'voigt':
            basis_matrix[:, i] = voigt(wavenumber_axis, center, gamma, gamma/2)
        else:
            raise ValueError(f"Unknown function type: {function_type}")
    
    return basis_matrix


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
    from scipy.signal import savgol_filter
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
