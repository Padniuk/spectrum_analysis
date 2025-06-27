import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


class Fitter:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.y_min, self.y_max = np.min(y), np.max(y)

    def fit(self, func, p0=None, bounds=(-np.inf, np.inf)):
        try:
            popt, _ = curve_fit(
                func, self.x, self.y, p0=p0, bounds=bounds, maxfev=100000
            )
        except (RuntimeError, ValueError):
            popt = np.zeros(
                len(p0) if p0 is not None else func.__code__.co_argcount - 1
            )
        return popt


class SignalFitter(Fitter):
    def __init__(self, x, y, sign=None):
        super().__init__(x, y)
        if sign is None:
            self.sign = self.define_sign(y)
        else:
            self.sign = sign

    def define_sign(self, signal):
        max_ = np.max(np.gradient(signal))
        min_ = np.min(np.gradient(signal))
        return 1 if max_ > abs(min_) else -1

    def sigmoid(self, x, amp, cen, wid, const):
        return amp / (1 + np.exp(-1 * self.sign * (x - cen) / wid)) + const

    def double_sigmoid(self, x, amp1, cen1, wid1, amp2, cen2, wid2, const):
        return (
            self.sigmoid(x, amp1, cen1, wid1, 0)
            + self.sigmoid(x, amp2, cen2, wid2, 0)
            + const
        )

    def one_exponent(self, x, amp, wid, cen=0, offset=0):
        return self.sign * amp * (1 - np.exp(-wid * (x - cen))) + offset

    def auto_borders(self, x, y):
        smooth_y = savgol_filter(y, 40, 2)
        derivative = np.gradient(smooth_y)
        derivative = savgol_filter(derivative, 400, 2)
        derivative = np.abs(derivative)
        threshold = 0.2 * np.max(derivative)

        time = np.array(x)
        time_points = [
            time[i]
            for i in range(len(derivative))
            if derivative[i] > threshold
            and time[i] < 0.7 * np.max(time)
            and time[i] > 0.3 * np.min(time)
        ]

        if len(time_points) < 2:
            return 0, 0
        left = time_points[0]
        right = time_points[-1]

        filtered_indices = np.where((time >= left) & (time <= right))[0]

        self.x = np.array(self.x)[filtered_indices]
        self.y = np.array(self.y)[filtered_indices]

        return left, right

    def fit_fast_component(self, func):
        popt = super().fit(
            func,
            p0=[
                0.5 * abs(np.max(self.y) + np.min(self.y)),
                0.5 * (self.x[0] + self.x[-1]),
                0.05,
                np.min(self.y),
            ],
            bounds=(
                [0, self.x[0], 0.000001, -np.inf],
                [abs(np.max(self.y) - np.min(self.y)), self.x[-1], np.inf, np.inf],
            ),
        )
        return np.append(popt, self.sign)

    def fit_slow_component(self, new_time, new_signals, right, signal_popt):
        if self.sign == 1:
            offset = signal_popt[3] + signal_popt[0]
        else:
            offset = signal_popt[3]
        func = lambda x, amp, wid: self.one_exponent(x, amp, wid, right, offset)
        try:
            popt, _ = curve_fit(
                func,
                new_time,
                new_signals,
                p0=[0.1, 0],
                bounds=([0, 0], [signal_popt[0], np.inf]),
            )
        except RuntimeError:
            popt = [0, 0]
        if (
            popt[0] < 0.1
            and self.sign == 1
            and signal_popt[0] + signal_popt[3] > np.mean(new_signals)
        ):
            popt[0] = np.mean(new_signals) - (signal_popt[0] + signal_popt[3])
            popt[1] = 0

        if popt[0] < 0.1 and self.sign == -1 and np.mean(new_signals) > signal_popt[3]:
            popt[0] = signal_popt[3] - np.mean(new_signals)
            popt[1] = 0

        if abs(signal_popt[0] + popt[0]) / (self.y_max - self.y_min) < 0.5:
            popt[0] = np.nan

        return popt

    def fast_rise_time(self):
        try:
            smooth_y = savgol_filter(self.y, 10, 2)
            derivative = np.gradient(smooth_y)
            max_derivative_idx = np.argmax(derivative)
            right_time = self.x[max_derivative_idx]

            growth_start = max_derivative_idx
            while (
                growth_start > 1
                and derivative[growth_start - 1] < derivative[growth_start]
            ):
                growth_start -= 1
            left_time = self.x[growth_start]
        except ValueError:
            return 0
        return right_time - left_time


class TriggerFitter(Fitter):
    def __init__(self, x, y):
        super().__init__(x, y)

    def gaussian(self, x, amp, cen, wid, bg):
        return amp * np.exp(-((x - cen) ** 2) / 2 / wid / wid) + bg

    def gauss(self, x, amp, cen, wid):
        return amp * np.exp(-((x - cen) ** 2) / 2 / wid / wid)
