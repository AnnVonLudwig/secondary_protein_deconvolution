"""
Peak Detection Module
====================

This module provides various peak detection algorithms for spectral analysis.
Includes methods for detecting sharp peaks, broad peaks, plateaus, and gradual increases.
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks


def detect_peaks(y, x_axis, threshold=0.005, window=1, min_distance=1):
    """
    Improved peak detection: derivative + curvature + local maximum + threshold
    
    Parameters:
    -----------
    y : array-like
        Spectral data
    x_axis : array-like
        Wavenumber axis
    threshold : float, default=0.005
        Intensity threshold (relative to maximum value)
    window : int, default=1
        Local maximum search window
    min_distance : int, default=1
        Minimum distance between peaks
    
    Returns:
    --------
    peaks : array
        Indices of detected peaks
    y_smooth : array
        Smoothed spectral data
    """
    # Smoothing - reduce smoothing to preserve more details
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    # Calculate derivatives
    dy = np.gradient(y_smooth)
    d2y = np.gradient(dy)
    
    # Set dynamic threshold - lower threshold to detect weak peaks
    max_intensity = np.max(y_smooth)
    dynamic_threshold = threshold * max_intensity
    
    # Initial candidate peaks - relax conditions to detect more peaks
    candidates = []
    for i in range(1, len(y_smooth)-1):
        # Conditions: first derivative crosses zero (from positive to negative) and second derivative is negative, and intensity exceeds threshold
        if (dy[i-1] > 0 and dy[i+1] < 0 and 
            d2y[i] < 0 and y_smooth[i] > dynamic_threshold):
            candidates.append(i)
    
    # If no peaks detected, further lower threshold
    if len(candidates) == 0:
        dynamic_threshold = threshold * 0.5 * max_intensity
        for i in range(1, len(y_smooth)-1):
            if (dy[i-1] > 0 and dy[i+1] < 0 and 
                d2y[i] < 0 and y_smooth[i] > dynamic_threshold):
                candidates.append(i)
    
    # Local maximum filtering - use smaller window
    filtered = []
    for idx in candidates:
        left = max(0, idx-window)
        right = min(len(y_smooth), idx+window)
        if y_smooth[idx] == np.max(y_smooth[left:right]):
            filtered.append(idx)
    
    # Distance filtering - reduce minimum distance
    final = []
    last_idx = -min_distance
    for idx in filtered:
        if idx - last_idx >= min_distance:
            final.append(idx)
            last_idx = idx
    
    return np.array(final, dtype=int), y_smooth


def detect_peaks_sensitive(y, x_axis, prominence=0.0001, distance=10):
    """
    More sensitive peak detection: using scipy.signal.find_peaks
    
    Parameters:
    -----------
    y : array-like
        Spectral data
    x_axis : array-like
        Wavenumber axis
    prominence : float, default=0.0001
        Peak prominence (minimum height relative to background)
    distance : int, default=10
        Minimum distance between peaks
    
    Returns:
    --------
    peaks : array
        Indices of detected peaks
    y_smooth : array
        Smoothed spectral data
    """
    # Smoothing
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    # Use find_peaks to detect peaks
    peaks, properties = find_peaks(y_smooth, prominence=prominence, distance=distance)
    
    return peaks, y_smooth


def detect_broad_peaks(y, x_axis, window_size=20, threshold_ratio=0.1):
    """
    Detect gradual/broad peaks: using sliding window method
    
    Parameters:
    -----------
    y : array-like
        Spectral data
    x_axis : array-like
        Wavenumber axis
    window_size : int, default=20
        Sliding window size
    threshold_ratio : float, default=0.1
        Threshold ratio relative to maximum value
    
    Returns:
    --------
    peaks : array
        Indices of detected peaks
    y_smooth : array
        Smoothed spectral data
    """
    # Smoothing
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    # Calculate local maximum within sliding window
    peaks = []
    threshold = threshold_ratio * np.max(y_smooth)
    
    for i in range(window_size//2, len(y_smooth) - window_size//2):
        window = y_smooth[i - window_size//2:i + window_size//2]
        if y_smooth[i] == np.max(window) and y_smooth[i] > threshold:
            # Check if too close to already detected peaks
            too_close = False
            for existing_peak in peaks:
                if abs(i - existing_peak) < window_size//2:
                    too_close = True
                    break
            if not too_close:
                peaks.append(i)
    
    return np.array(peaks, dtype=int), y_smooth


def detect_plateau_peaks(y, x_axis, min_width=10, threshold_ratio=0.02):
    """
    Detect plateau/gradual slopes: find relatively flat plateau regions
    
    Parameters:
    -----------
    y : array-like
        Spectral data
    x_axis : array-like
        Wavenumber axis
    min_width : int, default=10
        Minimum plateau width
    threshold_ratio : float, default=0.02
        Threshold ratio relative to maximum value
    
    Returns:
    --------
    peaks : array
        Indices of detected peaks
    y_smooth : array
        Smoothed spectral data
    """
    # Smoothing
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    peaks = []
    threshold = threshold_ratio * np.max(y_smooth)
    
    # Ensure min_width is at least 6 to avoid indexing issues
    min_width = max(6, min_width)
    half_width = min_width // 2
    
    # Find plateau regions
    for i in range(half_width, len(y_smooth) - half_width):
        # Check if current point is in plateau region
        left_window = y_smooth[i - half_width:i]
        right_window = y_smooth[i:i + half_width]
        
        # Calculate average of left and right windows
        left_avg = np.mean(left_window)
        right_avg = np.mean(right_window)
        current_val = y_smooth[i]
        
        # Plateau condition: current value above threshold, and both left and right window averages are below current value
        if (current_val > threshold and 
            current_val > left_avg * 1.05 and 
            current_val > right_avg * 1.05):
            
            # Check if too close to already detected peaks
            too_close = False
            for existing_peak in peaks:
                if abs(i - existing_peak) < min_width:
                    too_close = True
                    break
            if not too_close:
                peaks.append(i)
    
    return np.array(peaks, dtype=int), y_smooth


def detect_all_peaks_aggressive(y, x_axis):
    """
    Aggressive all-peak detection: using multiple method combination
    
    Parameters:
    -----------
    y : array-like
        Spectral data
    x_axis : array-like
        Wavenumber axis
    
    Returns:
    --------
    peaks : array
        Indices of detected peaks
    y_smooth : array
        Smoothed spectral data
    """
    # Smoothing
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    all_peaks = []
    
    # Method 1: find_peaks with very low prominence
    peaks1, _ = find_peaks(y_smooth, prominence=0.00001, distance=5)
    all_peaks.extend(peaks1)
    
    # Method 2: find all local maxima
    for i in range(2, len(y_smooth)-2):
        if (y_smooth[i] > y_smooth[i-1] and y_smooth[i] > y_smooth[i-2] and
            y_smooth[i] > y_smooth[i+1] and y_smooth[i] > y_smooth[i+2]):
            all_peaks.append(i)
    
    # Method 3: find relative high points (relative to surrounding regions)
    for i in range(10, len(y_smooth)-10):
        window = y_smooth[i-10:i+10]
        if y_smooth[i] > np.mean(window) * 1.02:  # 2% higher than surrounding average
            all_peaks.append(i)
    
    # Remove duplicates and peaks that are too close
    unique_peaks = []
    for peak in all_peaks:
        too_close = False
        for existing in unique_peaks:
            if abs(peak - existing) < 5:  # Minimum distance of 5 data points
                too_close = True
                break
        if not too_close:
            unique_peaks.append(peak)
    
    return np.array(unique_peaks, dtype=int), y_smooth


def detect_gradual_increase(y, x_axis, window_size=20, min_increase=0.0005):
    """
    Detect gradual slopes/progressive increase: specifically detect gently rising regions
    
    Parameters:
    -----------
    y : array-like
        Spectral data
    x_axis : array-like
        Wavenumber axis
    window_size : int, default=20
        Detection window size
    min_increase : float, default=0.0005
        Minimum increase amplitude
    
    Returns:
    --------
    peaks : array
        Indices of detected peaks
    y_smooth : array
        Smoothed spectral data
    """
    # Smoothing
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    peaks = []
    
    # Find progressively rising regions
    for i in range(window_size//2, len(y_smooth) - window_size//2):
        # Calculate average of front and back windows
        left_window = y_smooth[i - window_size//2:i]
        right_window = y_smooth[i:i + window_size//2]
        
        left_avg = np.mean(left_window)
        right_avg = np.mean(right_window)
        current_val = y_smooth[i]
        
        # Detect progressive increase: current value significantly higher than left, and right also higher than left
        if (current_val > left_avg + min_increase and 
            right_avg > left_avg + min_increase * 0.5):
            
            # Check if too close to already detected peaks
            too_close = False
            for existing_peak in peaks:
                if abs(i - existing_peak) < window_size//2:
                    too_close = True
                    break
            if not too_close:
                peaks.append(i)
    
    return np.array(peaks, dtype=int), y_smooth


def detect_manual_peaks(y, x_axis, manual_positions):
    """
    Manually specify peak positions
    
    Parameters:
    -----------
    y : array-like
        Spectral data
    x_axis : array-like
        Wavenumber axis
    manual_positions : list
        List of manually specified wavenumber positions
    
    Returns:
    --------
    peaks : array
        Indices of detected peaks
    y_smooth : array
        Smoothed spectral data
    """
    # Smoothing
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    peaks = []
    for pos in manual_positions:
        # Find index closest to specified position
        idx = np.abs(x_axis - pos).argmin()
        peaks.append(idx)
    
    return np.array(peaks, dtype=int), y_smooth


def combine_peaks(peak_arrays, min_distance=8):
    """
    Combine multiple peak detection results and remove duplicates
    
    Parameters:
    -----------
    peak_arrays : list of arrays
        List of peak index arrays from different detection methods
    min_distance : int, default=8
        Minimum distance between peaks to avoid duplicates
    
    Returns:
    --------
    unique_peaks : array
        Combined and deduplicated peak indices
    """
    # Combine all detected peaks
    all_peaks = np.concatenate(peak_arrays)
    
    # Remove duplicate peaks (peaks that are too close)
    unique_peaks = []
    for peak in all_peaks:
        too_close = False
        for existing in unique_peaks:
            if abs(peak - existing) < min_distance:
                too_close = True
                break
        if not too_close:
            unique_peaks.append(peak)
    
    return np.array(unique_peaks, dtype=int) 


def detect_gradual_decrease(y, x_axis, window_size=20, min_decrease=0.0005):
    """
    Detect gradual slopes/progressive decrease: specifically detect gently falling regions
    
    Parameters:
    -----------
    y : array-like
        Spectral data
    x_axis : array-like
        Wavenumber axis
    window_size : int, default=20
        Detection window size
    min_decrease : float, default=0.0005
        Minimum decrease amplitude
    
    Returns:
    --------
    peaks : array
        Indices of detected peaks
    y_smooth : array
        Smoothed spectral data
    """
    # Smoothing
    y_smooth = savgol_filter(y, window_length=9, polyorder=3)
    
    peaks = []
    
    # Find progressively falling regions
    for i in range(window_size//2, len(y_smooth) - window_size//2):
        # Calculate average of front and back windows
        left_window = y_smooth[i - window_size//2:i]
        right_window = y_smooth[i:i + window_size//2]
        
        left_avg = np.mean(left_window)
        right_avg = np.mean(right_window)
        current_val = y_smooth[i]
        
        # Detect progressive decrease: current value significantly lower than left, and right also lower than left
        if (current_val < left_avg - min_decrease and 
            right_avg < left_avg - min_decrease * 0.5):
            
            # Check if too close to already detected peaks
            too_close = False
            for existing_peak in peaks:
                if abs(i - existing_peak) < window_size//2:
                    too_close = True
                    break
            if not too_close:
                peaks.append(i)
    
    return np.array(peaks, dtype=int), y_smooth


def detect_all_slopes(y, x_axis, window_size=20, min_change=0.0005):
    """
    Detect both gradual increases and decreases
    
    Parameters:
    -----------
    y : array-like
        Spectral data
    x_axis : array-like
        Wavenumber axis
    window_size : int, default=20
        Detection window size
    min_change : float, default=0.0005
        Minimum change amplitude
    
    Returns:
    --------
    peaks : array
        Indices of detected peaks (both increases and decreases)
    y_smooth : array
        Smoothed spectral data
    """
    # Get both increase and decrease peaks
    increase_peaks, y_smooth = detect_gradual_increase(y, x_axis, window_size, min_change)
    decrease_peaks, _ = detect_gradual_decrease(y, x_axis, window_size, min_change)
    
    # Combine both types of peaks
    all_peaks = np.concatenate([increase_peaks, decrease_peaks])
    
    # Remove duplicates
    unique_peaks = []
    for peak in all_peaks:
        too_close = False
        for existing in unique_peaks:
            if abs(peak - existing) < window_size//2:
                too_close = True
                break
        if not too_close:
            unique_peaks.append(peak)
    
    return np.array(unique_peaks, dtype=int), y_smooth 