"""
Spectral Functions Module
========================

This module provides mathematical functions for spectral analysis,
including Lorentzian functions and other spectral modeling utilities.
"""

import numpy as np


def lorentzian(x, x0, gamma):
    """
    Lorentzian function
    
    Parameters:
    -----------
    x : array-like
        Wavenumber axis
    x0 : float
        Peak center
    gamma : float
        Linewidth parameter (FWHM)
    
    Returns:
    --------
    y : array
        Lorentzian function values
    """
    return 1 / (1 + ((x - x0) / (gamma / 2))**2)


def gaussian(x, x0, sigma):
    """
    Gaussian function
    
    Parameters:
    -----------
    x : array-like
        Wavenumber axis
    x0 : float
        Peak center
    sigma : float
        Standard deviation
    
    Returns:
    --------
    y : array
        Gaussian function values
    """
    return np.exp(-((x - x0) / sigma)**2 / 2)


def voigt(x, x0, gamma, sigma):
    """
    Voigt function (convolution of Lorentzian and Gaussian)
    
    Parameters:
    -----------
    x : array-like
        Wavenumber axis
    x0 : float
        Peak center
    gamma : float
        Lorentzian linewidth
    sigma : float
        Gaussian standard deviation
    
    Returns:
    --------
    y : array
        Voigt function values
    """
    # Simple approximation of Voigt function
    lorentz = lorentzian(x, x0, gamma)
    gauss = gaussian(x, x0, sigma)
    return (lorentz + gauss) / 2


def build_basis_functions(wavenumber_axis, peak_centers, peak_gammas, function_type='lorentzian'):
    """
    Build basis functions for spectral decomposition
    
    Parameters:
    -----------
    wavenumber_axis : array-like
        Wavenumber axis
    peak_centers : array-like
        Peak center positions
    peak_gammas : array-like
        Linewidth parameters for each peak
    function_type : str, default='lorentzian'
        Type of function to use ('lorentzian', 'gaussian', 'voigt')
    
    Returns:
    --------
    B : array
        Basis function matrix (n_points x n_peaks)
    """
    n_points = len(wavenumber_axis)
    n_peaks = len(peak_centers)
    
    B = np.zeros((n_points, n_peaks))
    
    for i, (center, gamma) in enumerate(zip(peak_centers, peak_gammas)):
        if function_type == 'lorentzian':
            B[:, i] = lorentzian(wavenumber_axis, center, gamma)
        elif function_type == 'gaussian':
            B[:, i] = gaussian(wavenumber_axis, center, gamma)
        elif function_type == 'voigt':
            B[:, i] = voigt(wavenumber_axis, center, gamma, gamma/2)
        else:
            raise ValueError(f"Unknown function type: {function_type}")
    
    # Normalize basis functions
    B /= B.max(axis=0, keepdims=True)
    
    return B


def normalize_spectrum(spectrum, method='max'):
    """
    Normalize spectrum using different methods
    
    Parameters:
    -----------
    spectrum : array-like
        Spectral data
    method : str, default='max'
        Normalization method ('max', 'area', 'min_max')
    
    Returns:
    --------
    normalized_spectrum : array
        Normalized spectrum
    """
    spectrum = np.array(spectrum)
    
    if method == 'max':
        return spectrum / np.max(spectrum)
    elif method == 'area':
        return spectrum / np.trapz(spectrum)
    elif method == 'min_max':
        return (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def smooth_spectrum(spectrum, window_length=9, polyorder=3):
    """
    Smooth spectrum using Savitzky-Golay filter
    
    Parameters:
    -----------
    spectrum : array-like
        Spectral data
    window_length : int, default=9
        Window length for smoothing
    polyorder : int, default=3
        Polynomial order for smoothing
    
    Returns:
    --------
    smoothed_spectrum : array
        Smoothed spectrum
    """
    from scipy.signal import savgol_filter
    return savgol_filter(spectrum, window_length=window_length, polyorder=polyorder)


def calculate_spectral_derivatives(spectrum, order=1):
    """
    Calculate spectral derivatives
    
    Parameters:
    -----------
    spectrum : array-like
        Spectral data
    order : int, default=1
        Order of derivative (1 or 2)
    
    Returns:
    --------
    derivatives : array or tuple
        First and/or second derivatives
    """
    if order == 1:
        return np.gradient(spectrum)
    elif order == 2:
        dy = np.gradient(spectrum)
        return np.gradient(dy)
    else:
        raise ValueError("Order must be 1 or 2")


def baseline_correction(spectrum, method='polynomial', degree=2):
    """
    Perform baseline correction
    
    Parameters:
    -----------
    spectrum : array-like
        Spectral data
    method : str, default='polynomial'
        Baseline correction method ('polynomial', 'linear')
    degree : int, default=2
        Polynomial degree for polynomial baseline
    
    Returns:
    --------
    corrected_spectrum : array
        Baseline-corrected spectrum
    """
    spectrum = np.array(spectrum)
    x = np.arange(len(spectrum))
    
    if method == 'polynomial':
        # Fit polynomial to spectrum
        coeffs = np.polyfit(x, spectrum, degree)
        baseline = np.polyval(coeffs, x)
    elif method == 'linear':
        # Linear baseline from first to last point
        baseline = np.linspace(spectrum[0], spectrum[-1], len(spectrum))
    else:
        raise ValueError(f"Unknown baseline correction method: {method}")
    
    return spectrum - baseline


def calculate_snr(spectrum, noise_region=None):
    """
    Calculate signal-to-noise ratio
    
    Parameters:
    -----------
    spectrum : array-like
        Spectral data
    noise_region : tuple, optional
        (start_index, end_index) for noise region calculation
    
    Returns:
    --------
    snr : float
        Signal-to-noise ratio
    """
    spectrum = np.array(spectrum)
    
    if noise_region is None:
        # Use first 10% of spectrum as noise region
        noise_region = (0, len(spectrum) // 10)
    
    start, end = noise_region
    noise = spectrum[start:end]
    noise_std = np.std(noise)
    
    if noise_std == 0:
        return float('inf')
    
    signal_max = np.max(spectrum)
    snr = signal_max / noise_std
    
    return snr 