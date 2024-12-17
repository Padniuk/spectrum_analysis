from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
from tools import TriggerFitter, SignalFitter
from tools import Validator
from utils import file_reader


class Plotter:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def plot_spectra(
        self,
        left_lim=0,
        right_lim=100,
        time_disc_left=0,
        time_disc_right=2,
        peak_left=0,
        peak_right=100,
    ):
        with open(os.path.join(self.folder_path, "tmp/signal.txt"), "r") as file:
            signal = file.readlines()
        with open(os.path.join(self.folder_path, "tmp/rise_time.txt"), "r") as file:
            rise_time = file.readlines()

        fast_amplitudes = [
            float(signal[i].split(",")[0])
            for i in Validator(self.folder_path).validated_indices()
        ]
        slow_amplitudes = [
            float(signal[i].split(",")[4])
            for i in Validator(self.folder_path).validated_indices()
        ]
        rise_times = [
            float(rise_time[i].split(",")[0])
            for i in Validator(self.folder_path).validated_indices()
        ]
        amplitudes = [
            fast_amplitude + slow_amplitude
            for fast_amplitude, slow_amplitude in zip(fast_amplitudes, slow_amplitudes)
        ]
        amplitudes = [
            amplitude
            for amplitude, rise_time in zip(amplitudes, rise_times)
            if rise_time > time_disc_left and rise_time < time_disc_right
        ]
        plt.hist(amplitudes, bins=60, range=(left_lim, right_lim))

        y, bin_edges = np.histogram(
            amplitudes,
            bins=int(60 / (right_lim - left_lim) * (peak_right - peak_left)),
            range=(peak_left, peak_right),
        )
        x = (bin_edges[:-1] + bin_edges[1:]) / 2
        tr_fitter = TriggerFitter(x, y)
        popt = tr_fitter.fit(
            tr_fitter.gauss,
            p0=[100, 0.5 * (peak_right - peak_left), 3],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        )
        _, mu, sigma = popt
        plt.plot(x, tr_fitter.gauss(x, *popt), "k-", linewidth=2)
        plt.xlabel("Signal amplitude, [mV]")
        plt.ylabel("Counts")
        plt.legend([f"Resolution: {round(100*2.355*abs(sigma/mu),2)}%"])
        plt.savefig(f"{self.folder_path}/tmp/amplitude_spectrum.png")
        plt.close()

    def plot_rise_time(self, left_lim=0, right_lim=2):
        with open(os.path.join(self.folder_path, "tmp/rise_time.txt"), "r") as file:
            rise_time = file.readlines()

        rise_time = [
            float(rise_time[i].split(",")[0])
            for i in Validator(self.folder_path).validated_indices()
        ]
        plt.hist(rise_time, bins=100, range=(left_lim, right_lim), alpha=0.75)
        max_bin_index = np.argmax(
            np.histogram(rise_time, bins=100, range=(left_lim, right_lim))[0]
        )
        bin_edges = np.histogram_bin_edges(
            rise_time, bins=100, range=(left_lim, right_lim)
        )
        max_bin_center = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
        plt.axvline(max_bin_center, color="r", linestyle="dashed", linewidth=1)
        plt.xlabel(r"Rise Time, [$\mu$s]")
        plt.legend([f"Max: {max_bin_center}"])
        plt.savefig(f"{self.folder_path}/tmp/rise_time.png")
        plt.close()

    def plot_rise_time_signal(
        self,
        left_lim_time=0,
        right_lim_time=2,
        left_lim_signal=0,
        right_lim_signal=100,
        is_log=False,
    ):
        with open(os.path.join(self.folder_path, "tmp/signal.txt"), "r") as file:
            signal = file.readlines()

        with open(os.path.join(self.folder_path, "tmp/rise_time.txt"), "r") as file:
            rise_time = file.readlines()

        signal_ampl = [
            float(signal[i].split(",")[0])
            for i in Validator(self.folder_path).validated_indices()
        ]
        slow_ampl = [
            float(signal[i].split(",")[4])
            for i in Validator(self.folder_path).validated_indices()
        ]
        rise_time = [
            float(rise_time[i].split(",")[0])
            for i in Validator(self.folder_path).validated_indices()
        ]
        total_signal = [signal_ampl[i] + slow_ampl[i] for i in range(len(signal_ampl))]

        if is_log:
            plt.hist2d(
                rise_time,
                total_signal,
                bins=(50, 50),
                range=[
                    [left_lim_time, right_lim_time],
                    [left_lim_signal, right_lim_signal],
                ],
                cmap=plt.cm.jet,
                norm=mpl.colors.LogNorm(),
            )
        else:
            plt.hist2d(
                rise_time,
                total_signal,
                bins=(50, 50),
                range=[
                    [left_lim_time, right_lim_time],
                    [left_lim_signal, right_lim_signal],
                ],
                cmap=plt.cm.jet,
            )

        plt.colorbar()
        plt.xlabel(r"Rise Time, [$\mu$s]")
        plt.ylabel("Signal Amplitude, [mV]")
        plt.savefig(f"{self.folder_path}/tmp/rise_time_signal.png")
        plt.close()

    def plot_sample(self, sample, left_lim=0, right_lim=-1):
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith(".csv")]

        time, signals, _ = file_reader(
            os.path.join(self.folder_path, csv_files[sample])
        )

        with open(os.path.join(self.folder_path, "tmp/signal.txt"), "r") as file:
            signal = file.readlines()

        popt = [float(par) for par in signal[sample].split(",")]
        if sample in list(Validator(self.folder_path).validated_indices()):
            plt.plot(time[left_lim:right_lim], signals[left_lim:right_lim])
            signal_fitter = SignalFitter(
                time, signals, Validator(self.folder_path).get_main_sign()
            )
            plt.plot(
                time[left_lim:right_lim],
                signal_fitter.sigmoid(time, *popt[:4])[left_lim:right_lim],
            )
            left, right = signal_fitter.auto_borders()

            if popt[-1] == 1:
                offset = popt[3] + popt[0]
            else:
                offset = popt[3]

            new_time = [t for t in time if t > right]

            plt.plot(
                new_time,
                signal_fitter.one_exponent(new_time, *popt[4:6], right, offset),
            )
            plt.axvline(right, color="r", linestyle="dashed", linewidth=2)
            plt.axvline(left, color="r", linestyle="dashed", linewidth=2)
            plt.xlabel(r"Time, [$\mu$s]")
            plt.ylabel("Signal Amplitude, [mV]")

            samples_folder = os.path.join(self.folder_path, "tmp/samples")
            os.makedirs(samples_folder, exist_ok=True)
            plt.savefig(os.path.join(samples_folder, f"{sample}.png"))
            plt.close()
