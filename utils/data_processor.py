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
        return {"trigger": [], "signal": [], "rise_time": []}
    if np.any(np.isinf(time)) or np.any(np.isinf(signal)) or np.any(np.isinf(trigger)):
        print(f"Inf values found in {file}")
        return {"trigger": [], "signal": [], "rise_time": []}
    if len(time) == 0 or len(signal) == 0 or len(trigger) == 0:
        print(f"Inf values found in {file} during reading")
        return {"trigger": [], "signal": [], "rise_time": []}

    tr_fitter = TriggerFitter(time, trigger)
    trigger_popt = tr_fitter.fit(tr_fitter.gaussian, p0=[0.5, 0, 1, 10])

    sig_fitter = SignalFitter(time, signal)
    try:
        signal = savgol_filter(signal, 40, 2)
    except ValueError:
        return {"trigger": [], "signal": [], "rise_time": []}
    left, right = sig_fitter.auto_borders(time, signal)
    if left == 0 and right == 0:
        print(f"Bad range values found in {file}")
        return {"trigger": [], "signal": [], "rise_time": []}
    fast_signal_popt = sig_fitter.fit_fast_component(sig_fitter.sigmoid)

    if fast_signal_popt[0] < 0.1:
        return {"trigger": [], "signal": [], "rise_time": []}
    new_time = [t for t in time if (t > right and t < 2*right - left)]

    new_indices = [i for i, t in enumerate(time) if (t > right and t < 2*right - left)]
    new_signals = signal[new_indices]
    new_signals = savgol_filter(new_signals, 200, 2)
    slow_signal_popt = sig_fitter.fit_slow_component(
        new_time, new_signals, right, fast_signal_popt
    )

    if np.isnan(slow_signal_popt[0]):
        return {"trigger": [], "signal": [], "rise_time": []}

    rise_time = -1 * fast_signal_popt[2] * np.log((1 - 0.9) / 0.9 * 0.1 / (1 - 0.1))
    return {
        "trigger": trigger_popt,
        "signal": [*(fast_signal_popt[:4]), *slow_signal_popt, fast_signal_popt[4]],
        "rise_time": [rise_time],
    }


def process_data(folder_path, num_workers):
    if num_workers > os.cpu_count() - 1:
        print("Number of workers exceeds the number of available threads")
        sys.exit(1)

    file_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]
    if not file_list:
        print("No CSV files found in the folder")
        sys.exit(1)

    print(f"Found {len(file_list)} files. Processing with {num_workers} workers...")

    with Pool(num_workers) as pool:
        result = pool.map(process_file, file_list)

    valid_results = [
        res
        for res in result
        if res["trigger"] is not None and res["signal"] is not None
    ]

    file_writer(folder_path, valid_results)

    print("Processing completed")
