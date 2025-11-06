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


def upper_envelope_smooth(y: np.ndarray, 
                         x: np.ndarray, 
                         peak_distance: int = 3, 
                         min_prominence_ratio: float = 0.02, 
                         local_window: int = 5) -> np.ndarray:
    """
    Upper envelope smoothing - creates a smooth curve that tightly follows the upper boundary
    of the data without protruding or curling up at the tail
    
    Parameters:
    -----------
    y : np.ndarray
        Intensity values
    x : np.ndarray
        Wavelength values
    peak_distance : int
        Minimum distance between peaks (smaller = more peaks, more details)
    min_prominence_ratio : float
        Minimum prominence ratio (relative to std) for peak detection
        (lower = more peaks detected)
    local_window : int
        Small window for local smoothing (preserves details)
        Must be odd number
        
    Returns:
    --------
    envelope : np.ndarray
        Upper envelope curve tightly following data peaks
    """
    from scipy.signal import find_peaks, savgol_filter
    from scipy.interpolate import UnivariateSpline
    from scipy.ndimage import maximum_filter1d
    
    # Step 1: Create initial envelope using running maximum to ensure it follows data closely
    # Use a small window to capture local peaks without over-smoothing
    window_size = max(3, local_window)
    if window_size % 2 == 0:
        window_size += 1
    
    # Running maximum filter - ensures envelope is always >= data
    envelope = maximum_filter1d(y, size=window_size, mode='reflect')
    
    # Step 2: Detect significant peaks to guide interpolation
    peak_prominence = np.std(y) * min_prominence_ratio
    peaks, _ = find_peaks(y, prominence=peak_prominence, distance=peak_distance)
    
    # Step 3: If we have enough peaks, use them for guided smoothing
    # Otherwise, use the running maximum as base and apply constrained smoothing
    if len(peaks) >= 3:
        # Use peak values as anchor points
        peak_indices = peaks
        peak_x = x[peak_indices]
        peak_y = y[peak_indices]  # Use original data, not envelope
        
        # Add boundary points at start and end
        if peak_indices[0] > 0:
            peak_x = np.concatenate([[x[0]], peak_x])
            peak_y = np.concatenate([[y[0]], peak_y])
        if peak_indices[-1] < len(x) - 1:
            peak_x = np.concatenate([peak_x, [x[-1]]])
            peak_y = np.concatenate([peak_y, [y[-1]]])
        
        # Use UnivariateSpline with smoothing factor that ensures it follows peaks closely
        # Lower s = tighter fit to points
        spline = UnivariateSpline(peak_x, peak_y, s=0, k=min(3, len(peak_x)-1))
        spline_envelope = spline(x)
        
        # Only use spline where it's >= original data and >= running max
        # This prevents protruding and ensures it follows data
        valid_mask = (spline_envelope >= y) & (spline_envelope >= envelope * 0.95)
        envelope[valid_mask] = np.minimum(spline_envelope[valid_mask], envelope[valid_mask] * 1.05)
    
    # Step 4: Apply constrained smoothing that never goes below original data
    if local_window >= 3 and local_window % 2 == 1:
        # Smooth the envelope
        envelope_smooth = savgol_filter(envelope, window_length=local_window, polyorder=2)
        
        # Only apply smoothing where:
        # 1. It doesn't go below original data
        # 2. It doesn't create artificial protrusions (limit upward deviation)
        valid_smooth = (envelope_smooth >= y) & (envelope_smooth <= envelope * 1.1)
        envelope[valid_smooth] = envelope_smooth[valid_smooth]
    
    # Step 5: Ensure envelope is always >= original data (critical constraint)
    envelope = np.maximum(envelope, y)
    
    # Step 6: Handle tail region specifically to prevent curling up
    # Check if tail region (last 20% of data) has low intensity
    tail_start_idx = int(len(y) * 0.8)
    tail_data = y[tail_start_idx:]
    tail_mean = np.mean(tail_data)
    tail_max = np.max(tail_data)
    data_mean = np.mean(y)
    data_max = np.max(y)
    
    # If tail is significantly lower than overall data, constrain envelope in tail region
    if tail_mean < data_mean * 0.3 or tail_max < data_max * 0.2:
        # In tail region, envelope should closely follow the data
        # Don't allow envelope to exceed local tail data by more than a small margin
        tail_max_allowed = tail_max * 1.15  # Allow only 15% above tail max
        
        # Find where envelope exceeds the allowed limit in tail
        tail_envelope = envelope[tail_start_idx:]
        excess_mask = tail_envelope > tail_max_allowed
        
        if np.any(excess_mask):
            # Smoothly reduce excessive values in tail region
            # Use a gradual approach: find the point where envelope starts to exceed
            transition_idx = tail_start_idx
            for i in range(tail_start_idx, len(envelope)):
                if envelope[i] > tail_max_allowed:
                    transition_idx = i
                    break
            
            # Apply smooth reduction from transition point to end
            if transition_idx < len(envelope) - 1:
                # Get the value just before transition as reference
                ref_value = envelope[transition_idx - 1] if transition_idx > 0 else envelope[transition_idx]
                # Target: don't exceed tail_max_allowed
                target_value = min(tail_max_allowed, ref_value)
                
                # Smoothly interpolate from transition to end
                for i in range(transition_idx, len(envelope)):
                    progress = (i - transition_idx) / max(1, len(envelope) - transition_idx)
                    # Gradually reduce from ref_value to target_value
                    allowed = ref_value * (1 - 0.5 * progress) + target_value * (0.5 * progress)
                    # Ensure it's at least the original data value
                    envelope[i] = min(envelope[i], max(y[i], allowed))
        
        # Final constraint: in tail region, envelope should be very close to data
        # Limit envelope to be within a small margin of the actual data in tail
        tail_envelope_clamped = np.minimum(envelope[tail_start_idx:], 
                                          tail_max_allowed)
        envelope[tail_start_idx:] = np.maximum(y[tail_start_idx:], tail_envelope_clamped)
    
    # Step 7: Final pass - ensure no artificial protrusions
    # Limit how much envelope can exceed local data maximum
    local_max_window = max(5, local_window)
    if local_max_window % 2 == 0:
        local_max_window += 1
    local_data_max = maximum_filter1d(y, size=local_max_window, mode='reflect')
    # Don't allow envelope to exceed local max by more than 10%
    envelope = np.minimum(envelope, local_data_max * 1.1)
    
    # Final constraint: always >= original data
    envelope = np.maximum(envelope, y)
    
    return envelope
