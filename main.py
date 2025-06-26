import argparse
from utils import process_data, Plotter
from configs import settings

def main(folder_path, num_workers=2, sample=None):
    if sample is None:
        process_data(folder_path, num_workers)
        Plotter(folder_path).plot_spectra(right_lim=settings.right_lim, left_lim=settings.left_lim, time_disc_left=settings.time_disc_left, time_disc_right=settings.time_disc_right, peak_left=settings.peak_left, peak_right=settings.peak_right)
        Plotter(folder_path).plot_rise_time(left_lim=settings.right_lim, right_lim=settings.right_lim)
        Plotter(folder_path).plot_rise_time_signal(
            left_lim_time=settings.time_disc_left, right_lim_time=settings.time_disc_right,
            left_lim_signal=settings.left_lim, right_lim_signal=settings.right_lim,
            is_log=settings.IS_LOG
        )
        if settings.all_peaks:
            Plotter(folder_path).plot_all_peaks()
    else:
        Plotter(folder_path).plot_sample(sample)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parameters")
    parser.add_argument("-t", "--threads", type=int, help="Number of workers")
    parser.add_argument("-f", "--folder", type=str, help="Folder path")
    parser.add_argument("-s", "--sample", type=int, help="Sample number")
    args = parser.parse_args()
    main(args.folder, args.threads, args.sample)
