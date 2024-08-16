import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob

def load_and_normalize_data(month_dir, min_val, max_val):
    """
    Load data and normalize it using provided min and max values.
    Returns normalized data and raw data lists.
    """
    nc_files = glob.glob(os.path.join(month_dir, '*.nc'))
    normalized_data = []
    raw_data = []
    for file_path in tqdm(nc_files, desc=f"Processing files in {month_dir}"):
        with xr.open_dataset(file_path) as data:
            normalized_vars = {}
            raw_vars = {}
            for variable in ['omega', 'temp', 'qv']:
                values = data[variable].values
                raw_vars[variable] = values.copy()
                normalized_vars[variable] = (values - min_val[variable]) / (max_val[variable] - min_val[variable])
            normalized_data.append(normalized_vars)
            raw_data.append(raw_vars)
    return normalized_data, raw_data

def plot_data_comparison(raw_data, normalized_data, variable, title, save_path):
    """Plot comparison of raw and normalized data."""
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    sns.histplot(raw_data.flatten(), kde=True, color='blue', label='Original Data')
    plt.title('Original ' + variable)
    plt.xlabel(variable)
    plt.ylabel('Density')
    plt.legend()
    
    plt.subplot(122)
    sns.histplot(normalized_data.flatten(), kde=True, color='green', label='Normalized Data')
    plt.title('Normalized ' + variable)
    plt.xlabel(variable)
    plt.ylabel('Density')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)  # Save the image
    plt.close()  # Close the figure window to avoid memory leaks

def main():
    root_dir = '/workspace/caiman_datasets'
    output_dir = '/workspace/project_caiman/normalized_data'
    month = 'Jan2011'
    min_val = {}
    max_val = {}
    for variable in ['omega', 'temp', 'qv']:
        min_val[variable] = np.load(os.path.join(output_dir, f'min_{variable}.npy'))
        max_val[variable] = np.load(os.path.join(output_dir, f'max_{variable}.npy'))
    
    month_dir = os.path.join(root_dir, month)
    normalized_data, raw_data = load_and_normalize_data(month_dir, min_val, max_val)
    
    for variable in ['omega', 'temp', 'qv']:
        raw_list = [data[variable] for data in raw_data]
        norm_list = [data[variable] for data in normalized_data]
        flat_raw = np.concatenate(raw_list)
        flat_normalized = np.concatenate(norm_list)
        save_path = os.path.join(output_dir, f'comparison_{variable}_{month}.png')  # Set save path
        plot_data_comparison(flat_raw, flat_normalized, variable, f'Comparison for {variable} in {month}', save_path)

if __name__ == '__main__':
    main()
