"""
Peak Classification Module
=========================

This module provides peak classification functionality for protein secondary structure analysis.
Includes classification tables and functions for identifying peaks based on wavenumber positions.
"""

import numpy as np


# Unified linewidth parameter - modify this single variable to adjust linewidth for all peaks
GAMMA_WIDTH = 12.0  # Linewidth parameter, can be adjusted to 8.0, 10.0, 15.0, 20.0, etc.

# Protein secondary structure peak classification table
# Format: (center wavenumber, tolerance range, linewidth parameter)
PEAK_TABLE = {
    "beta_sheet": [
        (1587, 5.0, GAMMA_WIDTH),  # Added peak near 1587
        (1624, 5.0, GAMMA_WIDTH),  # Expanded tolerance range
        (1627, 5.0, GAMMA_WIDTH),
        (1633, 5.0, GAMMA_WIDTH),
        (1638, 5.0, GAMMA_WIDTH),
        (1642, 5.0, GAMMA_WIDTH),
        (1691, 5.0, GAMMA_WIDTH),
        (1696, 5.0, GAMMA_WIDTH),
        (1704, 5.0, GAMMA_WIDTH),  # Added peak near 1704
    ],
    "alpha_helix": [
        (1656, 5.0, GAMMA_WIDTH),  # Expanded tolerance range
        (1664, 5.0, GAMMA_WIDTH),  # Added peak near 1664
    ],
    "beta_turn": [
        (1667, 5.0, GAMMA_WIDTH),  # Expanded tolerance range
        (1675, 5.0, GAMMA_WIDTH),
        (1680, 5.0, GAMMA_WIDTH),
        (1685, 5.0, GAMMA_WIDTH),
    ],
}


def classify_peak(peak_wavenumber, gamma_width=12.0, peak_table=None):
    """
    Classify peaks into protein secondary structure families based on wavenumber position
    
    Parameters:
    -----------
    peak_wavenumber : float
        Wavenumber position of the peak
    gamma_width : float, default=12.0
        Linewidth parameter for unknown peaks and to override default
    peak_table : dict, optional
        Peak classification table. If None, uses the default PEAK_TABLE
    
    Returns:
    --------
    family_name : str or None
        Name of the protein secondary structure family
    linewidth_parameter : float or None
        Linewidth parameter for the peak
    """
    if peak_table is None:
        # Create a copy of PEAK_TABLE with the specified gamma_width
        peak_table = {}
        for family, peaks in PEAK_TABLE.items():
            peak_table[family] = [(center, delta, gamma_width) for center, delta, _ in peaks]
    
    for family, peaks in peak_table.items():
        for center, delta, gamma in peaks:
            if abs(peak_wavenumber - center) <= delta:
                return family, gamma
    return None, None


def classify_peaks(peak_wavenumbers, gamma_width=12.0, peak_table=None):
    """
    Classify multiple peaks into protein secondary structure families
    
    Parameters:
    -----------
    peak_wavenumbers : array-like
        Array of wavenumber positions
    gamma_width : float, default=12.0
        Linewidth parameter for all peak types
    peak_table : dict, optional
        Peak classification table. If None, uses the default PEAK_TABLE
    
    Returns:
    --------
    families : array
        Array of family names (None for unknown peaks)
    linewidths : array
        Array of linewidth parameters for each peak
    """
    if peak_table is None:
        # Create a copy of PEAK_TABLE with the specified gamma_width
        peak_table = {}
        for family, peaks in PEAK_TABLE.items():
            peak_table[family] = [(center, delta, gamma_width) for center, delta, _ in peaks]
    
    families = []
    linewidths = []
    
    for wavenumber in peak_wavenumbers:
        family, linewidth = classify_peak(wavenumber, gamma_width, peak_table)
        if family is not None:
            families.append(family)
            linewidths.append(linewidth)
        else:
            families.append("unknown")
            linewidths.append(gamma_width)
    
    return np.array(families), np.array(linewidths)


def get_peak_statistics(peak_wavenumbers, peak_types, amplitudes):
    """
    Calculate statistics for detected peaks by structure type
    
    Parameters:
    -----------
    peak_wavenumbers : array-like
        Array of peak wavenumber positions
    peak_types : array-like
        Array of peak type classifications
    amplitudes : array-like
        Array of peak amplitudes
    
    Returns:
    --------
    stats : dict
        Dictionary containing statistics for each structure type
    """
    stats = {
        'beta_sheet': {'count': 0, 'intensity': 0.0, 'positions': []},
        'alpha_helix': {'count': 0, 'intensity': 0.0, 'positions': []},
        'beta_turn': {'count': 0, 'intensity': 0.0, 'positions': []},
        'unknown': {'count': 0, 'intensity': 0.0, 'positions': []}
    }
    
    for pos, peak_type, amp in zip(peak_wavenumbers, peak_types, amplitudes):
        if peak_type in stats:
            stats[peak_type]['count'] += 1
            stats[peak_type]['intensity'] += amp
            stats[peak_type]['positions'].append(pos)
    
    return stats


def calculate_structure_ratios(peak_types, amplitudes):
    """
    Calculate ratios between different protein secondary structures
    
    Parameters:
    -----------
    peak_types : array-like
        Array of peak type classifications
    amplitudes : array-like
        Array of peak amplitudes
    
    Returns:
    --------
    ratios : dict
        Dictionary containing various structure ratios
    """
    # Calculate total intensities for each structure type
    beta_intensity = sum(amp for amp, ptype in zip(amplitudes, peak_types) 
                        if ptype in ['beta_sheet', 'beta_turn'])
    alpha_intensity = sum(amp for amp, ptype in zip(amplitudes, peak_types) 
                         if ptype == 'alpha_helix')
    unknown_intensity = sum(amp for amp, ptype in zip(amplitudes, peak_types) 
                           if ptype == 'unknown')
    
    total_intensity = sum(amplitudes)
    
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


def print_peak_summary(peak_wavenumbers, peak_types, amplitudes):
    """
    Print a summary of detected peaks and their classifications
    
    Parameters:
    -----------
    peak_wavenumbers : array-like
        Array of peak wavenumber positions
    peak_types : array-like
        Array of peak type classifications
    amplitudes : array-like
        Array of peak amplitudes
    """
    print(f"All detected peaks: {len(peak_wavenumbers)}")
    
    for i, (pos, peak_type, amp) in enumerate(zip(peak_wavenumbers, peak_types, amplitudes)):
        if peak_type == "unknown":
            print(f"  Peak {i+1}: {pos:.1f} cm⁻¹ -> Unknown type - Amplitude: {amp:.6f}")
        else:
            print(f"  Peak {i+1}: {pos:.1f} cm⁻¹ -> {peak_type} - Amplitude: {amp:.6f}")
    
    # Calculate and print structure ratios
    ratios = calculate_structure_ratios(peak_types, amplitudes)
    
    print(f"\nStructure Analysis:")
    print(f"  Total β structure intensity: {ratios['beta_intensity']:.6f}")
    print(f"  α-helix intensity: {ratios['alpha_intensity']:.6f}")
    print(f"  Unknown type intensity: {ratios['unknown_intensity']:.6f}")
    
    if ratios['alpha_intensity'] > 0:
        print(f"  β/α ratio: {ratios['beta_alpha_ratio']:.3f}")
    else:
        print("  No α-helix structure detected")
    
    print(f"  β structure percentage: {ratios['beta_percentage']:.1f}%")
    print(f"  α-helix percentage: {ratios['alpha_percentage']:.1f}%")
    print(f"  Unknown percentage: {ratios['unknown_percentage']:.1f}%")


def get_peak_table():
    """
    Get the default peak classification table
    
    Returns:
    --------
    peak_table : dict
        The default peak classification table
    """
    return PEAK_TABLE.copy()


def set_gamma_width(gamma_width):
    """
    Set the global gamma width parameter for all peaks
    
    Parameters:
    -----------
    gamma_width : float
        New gamma width value
    """
    global GAMMA_WIDTH
    GAMMA_WIDTH = gamma_width 