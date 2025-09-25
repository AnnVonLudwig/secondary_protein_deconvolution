"""
比例计算模块
============

提供β/α比例计算功能。
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from .peak_detection import detect_peaks_basic
from .peak_classification import classify_peaks, calculate_structure_ratios
from config.analysis_config import INTEGRAL_ANALYSIS_CONFIG


@dataclass
class RatioResult:
    """比例计算结果"""
    method: str
    beta_alpha_ratio: float
    beta_intensity: float
    alpha_intensity: float
    beta_percentage: float
    alpha_percentage: float
    total_intensity: float
    confidence: float = 0.0
    details: Optional[Dict] = None


class RatioCalculator:
    """比例计算器"""
    
    def __init__(self):
        """初始化计算器"""
        self.peak_regions = INTEGRAL_ANALYSIS_CONFIG['peak_regions']
    
    def calculate_peak_based_ratio(self, spectrum: np.ndarray, 
                                 wavenumber_axis: np.ndarray) -> RatioResult:
        """
        使用峰值法计算β/α比例
        
        Parameters:
        -----------
        spectrum : np.ndarray
            光谱数据
        wavenumber_axis : np.ndarray
            波数轴
            
        Returns:
        --------
        RatioResult
            计算结果
        """
        try:
            # 检测峰值
            peak_indices, _ = detect_peaks_basic(spectrum, wavenumber_axis)
            
            if len(peak_indices) == 0:
                return RatioResult(
                    method="peak_based",
                    beta_alpha_ratio=0.0,
                    beta_intensity=0.0,
                    alpha_intensity=0.0,
                    beta_percentage=0.0,
                    alpha_percentage=0.0,
                    total_intensity=0.0,
                    confidence=0.0
                )
            
            # 获取峰值位置和强度
            peak_positions = wavenumber_axis[peak_indices]
            peak_amplitudes = spectrum[peak_indices]
            
            # 分类峰值
            peak_types, _ = classify_peaks(peak_positions)
            
            # 计算结构比例
            ratios = calculate_structure_ratios(peak_types, peak_amplitudes)
            
            # 计算置信度
            confidence = self._calculate_confidence(peak_amplitudes, len(peak_indices))
            
            return RatioResult(
                method="peak_based",
                beta_alpha_ratio=ratios['beta_alpha_ratio'],
                beta_intensity=ratios['beta_intensity'],
                alpha_intensity=ratios['alpha_intensity'],
                beta_percentage=ratios['beta_percentage'],
                alpha_percentage=ratios['alpha_percentage'],
                total_intensity=ratios['total_intensity'],
                confidence=confidence,
                details={
                    'peak_positions': peak_positions.tolist(),
                    'peak_amplitudes': peak_amplitudes.tolist(),
                    'peak_types': peak_types
                }
            )
            
        except Exception as e:
            return RatioResult(
                method="peak_based",
                beta_alpha_ratio=0.0,
                beta_intensity=0.0,
                alpha_intensity=0.0,
                beta_percentage=0.0,
                alpha_percentage=0.0,
                total_intensity=0.0,
                confidence=0.0
            )
    
    def calculate_integral_based_ratio(self, spectrum: np.ndarray, 
                                     wavenumber_axis: np.ndarray) -> RatioResult:
        """
        使用积分法计算β/α比例
        
        Parameters:
        -----------
        spectrum : np.ndarray
            光谱数据
        wavenumber_axis : np.ndarray
            波数轴
            
        Returns:
        --------
        RatioResult
            计算结果
        """
        try:
            # 计算各结构类型的积分面积
            alpha_area = self._calculate_structure_area(spectrum, wavenumber_axis, 
                                                      self.peak_regions['alpha_helix'])
            beta_sheet_area = self._calculate_structure_area(spectrum, wavenumber_axis, 
                                                           self.peak_regions['beta_sheet'])
            beta_turn_area = self._calculate_structure_area(spectrum, wavenumber_axis, 
                                                          self.peak_regions['beta_turn'])
            
            beta_total_area = beta_sheet_area + beta_turn_area
            total_area = alpha_area + beta_total_area
            
            # 计算比例
            if alpha_area > 1e-8:
                beta_alpha_ratio = beta_total_area / alpha_area
            else:
                beta_alpha_ratio = float('inf')
            
            # 计算百分比
            if total_area > 0:
                alpha_percentage = (alpha_area / total_area) * 100
                beta_percentage = (beta_total_area / total_area) * 100
            else:
                alpha_percentage = 0.0
                beta_percentage = 0.0
            
            # 计算置信度
            confidence = self._calculate_integral_confidence(spectrum, alpha_area, beta_total_area)
            
            return RatioResult(
                method="integral_based",
                beta_alpha_ratio=beta_alpha_ratio,
                beta_intensity=beta_total_area,
                alpha_intensity=alpha_area,
                beta_percentage=beta_percentage,
                alpha_percentage=alpha_percentage,
                total_intensity=total_area,
                confidence=confidence,
                details={
                    'alpha_area': alpha_area,
                    'beta_sheet_area': beta_sheet_area,
                    'beta_turn_area': beta_turn_area,
                    'beta_total_area': beta_total_area,
                    'total_area': total_area
                }
            )
            
        except Exception as e:
            return RatioResult(
                method="integral_based",
                beta_alpha_ratio=0.0,
                beta_intensity=0.0,
                alpha_intensity=0.0,
                beta_percentage=0.0,
                alpha_percentage=0.0,
                total_intensity=0.0,
                confidence=0.0
            )
    
    def _calculate_structure_area(self, spectrum: np.ndarray, 
                                wavenumber_axis: np.ndarray, 
                                regions: List[Tuple[float, float]]) -> float:
        """计算指定区域的光谱面积"""
        total_area = 0.0
        
        for start, end in regions:
            mask = (wavenumber_axis >= start) & (wavenumber_axis <= end)
            if np.any(mask):
                region_spectrum = spectrum[mask]
                region_wavenumber = wavenumber_axis[mask]
                area = np.trapz(region_spectrum, region_wavenumber)
                total_area += area
        
        return total_area
    
    def _calculate_confidence(self, amplitudes: np.ndarray, num_peaks: int) -> float:
        """计算峰值法的置信度"""
        if len(amplitudes) == 0:
            return 0.0
        
        # 基于峰值强度和数量的置信度
        intensity_factor = min(np.max(amplitudes) / np.mean(amplitudes), 10.0) / 10.0
        count_factor = min(num_peaks / 10.0, 1.0)
        
        return (intensity_factor + count_factor) / 2.0
    
    def _calculate_integral_confidence(self, spectrum: np.ndarray, 
                                     alpha_area: float, beta_area: float) -> float:
        """计算积分法的置信度"""
        if alpha_area + beta_area == 0:
            return 0.0
        
        # 基于信号强度和信噪比的置信度
        signal_strength = np.max(spectrum) - np.min(spectrum)
        noise_level = np.std(spectrum)
        snr = signal_strength / noise_level if noise_level > 0 else 0
        
        # 基于结构比例合理性的置信度
        ratio_reasonableness = 1.0 - abs(np.log10((beta_area / alpha_area) + 1e-8)) / 10.0
        ratio_reasonableness = max(0.0, min(1.0, ratio_reasonableness))
        
        snr_factor = min(snr / 100.0, 1.0)
        
        return (snr_factor + ratio_reasonableness) / 2.0


def compare_ratio_methods(spectrum: np.ndarray, wavenumber_axis: np.ndarray) -> Dict[str, RatioResult]:
    """
    对比不同β/α比例计算方法
    
    Parameters:
    -----------
    spectrum : np.ndarray
        光谱数据
    wavenumber_axis : np.ndarray
        波数轴
        
    Returns:
    --------
    dict
        包含不同方法结果的字典
    """
    calculator = RatioCalculator()
    
    results = {
        'peak_based': calculator.calculate_peak_based_ratio(spectrum, wavenumber_axis),
        'integral_based': calculator.calculate_integral_based_ratio(spectrum, wavenumber_axis)
    }
    
    return results
