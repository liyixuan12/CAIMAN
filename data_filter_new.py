import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def load_and_normalize_data(month_dir, min_val, max_val):
    """加载数据，并使用提供的最小最大值进行归一化。返回归一化数据列表。"""
    nc_files = glob.glob(os.path.join(month_dir, '*.nc'))
    normalized_data = []
    for file_path in nc_files:
        with xr.open_dataset(file_path) as data:
            normalized_vars = {}
            for variable in ['omega', 'temp', 'qv']:
                values = data[variable].values
                normalized_vars[variable] = (values - min_val[variable]) / (max_val[variable] - min_val[variable])
            file_name = os.path.basename(file_path)
            normalized_data.append((file_name, normalized_vars['omega'], normalized_vars['temp'], normalized_vars['qv']))
    return normalized_data

def save_normalization_params(output_dir, min_val, max_val):
    """保存最小和最大值到指定目录。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for variable in min_val.keys():
        np.save(os.path.join(output_dir, f'min_{variable}.npy'), min_val[variable])
        np.save(os.path.join(output_dir, f'max_{variable}.npy'), max_val[variable])

def main():
    root_dir = '/workspace/caiman_datasets'
    output_dir = '/workspace/project_caiman/normalized_data'
    train_months = ['Jan2011', 'Feb2011', 'March2011', 'April2011', 'May2011', 'June2011', 'July2011', 'August2011', 'September2011']
    test_months = ['October2011', 'November2011', 'December2011']

    # 初始化归一化参数
    min_val = {'omega': np.inf, 'temp': np.inf, 'qv': np.inf}
    max_val = {'omega': -np.inf, 'temp': -np.inf, 'qv': -np.inf}

    # 计算训练数据的最小最大值
    for month in train_months:
        month_dir = os.path.join(root_dir, month)
        for file_path in glob.glob(os.path.join(month_dir, '*.nc')):
            with xr.open_dataset(file_path) as data:
                for variable in ['omega', 'temp', 'qv']:
                    values = data[variable].values
                    min_val[variable] = np.minimum(min_val[variable], np.min(values))
                    max_val[variable] = np.maximum(max_val[variable], np.max(values))

    # 保存归一化参数
    save_normalization_params(output_dir, min_val, max_val)

    # 重新加载归一化参数
    min_val, max_val = {}, {}
    for variable in ['omega', 'temp', 'qv']:
        min_val[variable] = np.load(os.path.join(output_dir, f'min_{variable}.npy'))
        max_val[variable] = np.load(os.path.join(output_dir, f'max_{variable}.npy'))

    # 归一化数据并直接使用，不保存
    for month in train_months:
        month_dir = os.path.join(root_dir, month)
        normalized_data = load_and_normalize_data(month_dir, min_val, max_val)
        #save_normalized_data(output_dir, normalized_data)
    print("Normalized data range check:")
    for variable in ['omega', 'temp', 'qv']:
        normalized_min = np.min(normalized_data[variable])
        normalized_max = np.max(normalized_data[variable])
        print(f"{variable.capitalize()} - Min: {normalized_min}, Max: {normalized_max}")
        assert 0 <= normalized_min <= 1 and 0 <= normalized_max <= 1, "Data not properly normalized"

    
      # 归一化测试数据并保存
    for month in test_months:
        month_dir = os.path.join(root_dir, month)
        normalized_data = load_and_normalize_data(month_dir, min_val, max_val)
        #save_normalized_data(output_dir, normalized_data)

    
    print(min_val, max_val)
        # 这里可以根据需要使用 normalized_data，例如打印或进一步分析

if __name__ == '__main__':
    main()

