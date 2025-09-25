"""
数据加载器
==========

提供各种数据加载功能。
"""

import numpy as np
import os
from typing import Tuple, Dict, Optional


def load_spectral_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载光谱数据
    
    Parameters:
    -----------
    file_path : str
        数据文件路径
        
    Returns:
    --------
    wavenumber_axis : np.ndarray
        波数轴
    spectrum : np.ndarray
        光谱数据
    """
    # 根据文件扩展名选择加载方法
    if file_path.endswith('.npz'):
        return load_npz_data(file_path)
    elif file_path.endswith('.mat'):
        return load_mat_data(file_path)
    elif file_path.endswith('.txt'):
        return load_txt_data(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def load_npz_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载NPZ格式数据"""
    data = np.load(file_path)
    # 根据实际数据结构调整
    if 'wavenumber_axis' in data:
        return data['wavenumber_axis'], data['composite']
    else:
        # 默认处理
        keys = list(data.keys())
        return data[keys[0]], data[keys[1]]


def load_mat_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载MAT格式数据"""
    from scipy.io import loadmat
    data = loadmat(file_path)
    # 根据实际数据结构调整
    return data['wavenumber_axis'], data['spectrum']


def load_txt_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载TXT格式数据"""
    data = np.loadtxt(file_path)
    wavenumber = data[:, 0]
    spectrum = data[:, 1]
    return wavenumber, spectrum


def load_batch_data(data_dir: str) -> Dict[str, np.ndarray]:
    """
    批量加载数据
    
    Parameters:
    -----------
    data_dir : str
        数据目录路径
        
    Returns:
    --------
    data_dict : dict
        包含所有数据的字典
    """
    data_dict = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith(('.npz', '.mat', '.txt')):
            file_path = os.path.join(data_dir, filename)
            try:
                wavenumber, spectrum = load_spectral_data(file_path)
                data_dict[filename] = {
                    'wavenumber': wavenumber,
                    'spectrum': spectrum
                }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return data_dict
