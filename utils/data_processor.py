from multiprocessing import Pool
import sys
from tools import SignalFitter, TriggerFitter
import os
import numpy as np
from utils import file_reader, file_writer
from scipy.signal import savgol_filter

def process_file(file):
    time, signal, trigger = file_reader(file)
    
    if np.any(np.isnan(time)) or np.any(np.isnan(signal)) or np.any(np.isnan(trigger)):
        print(f"NaN values found in {file}")
        return {'trigger': [], 'signal': [], 'rise_time': []}
    if np.any(np.isinf(time)) or np.any(np.isinf(signal)) or np.any(np.isinf(trigger)):
        print(f"Inf values found in {file}")
        return {'trigger': [], 'signal': [], 'rise_time': []}

    tr_fitter = TriggerFitter(time, trigger)
    trigger_popt = tr_fitter.fit(tr_fitter.gaussian, p0=[0.5, 0, 1, 10])
    
    signal = savgol_filter(signal, 40, 2)
    sig_fitter = SignalFitter(time, signal)
    sig_fitter.auto_borders()
    signal_popt = sig_fitter.fit(sig_fitter.sigmoid)

    rise_time = -1*signal_popt[2]*np.log((1-0.9)/0.9*0.1/(1-0.1))
    return {'trigger': trigger_popt, 'signal': signal_popt, 'rise_time': [rise_time]}

def process_data(folder_path, num_workers):
    if num_workers > os.cpu_count() - 1:
        print("Number of workers exceeds the number of available threads")
        sys.exit(1)

    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not file_list:
        print("No CSV files found in the folder")
        sys.exit(1)

    print(f"Found {len(file_list)} files. Processing with {num_workers} workers...")

    with Pool(num_workers) as pool:
        result = pool.map(process_file, file_list)

    valid_results = [res for res in result if res['trigger'] is not None and res['signal'] is not None]
    
    file_writer(folder_path, valid_results)

    print("Processing completed")