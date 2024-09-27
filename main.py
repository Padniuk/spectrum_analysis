import os
import argparse
import matplotlib.pyplot as plt
from utils import file_reader, process_data, Plotter
from tools import Validator
from scipy.signal import savgol_filter

def main(folder_path, num_workers=2, sample=None):
    if sample is None:
        process_data(folder_path, num_workers)
    else:
        Plotter(folder_path).plot_spectra()
        Plotter(folder_path).plot_rise_time()
        Plotter(folder_path).plot_rise_time_signal()
        Plotter(folder_path).plot_sample(sample)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process parameters')
    parser.add_argument('-t', '--threads', type=int, help='Number of workers')
    parser.add_argument('-f', '--folder', type=str, help='Folder path')
    parser.add_argument('-s', '--sample', type=int, help='Sample number')
    args = parser.parse_args()
    main(args.folder, args.threads, args.sample)
