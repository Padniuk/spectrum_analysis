import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

class Fitter:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit(self, func, p0=None, bounds=(-np.inf, np.inf)):
        try:
            popt, _ = curve_fit(func, self.x, self.y, p0=p0, bounds=bounds, maxfev=100000)
        except (RuntimeError, ValueError):
            popt = np.zeros(len(p0) if p0 is not None else func.__code__.co_argcount - 1)
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
        return amp / (1 + np.exp(-1*self.sign*(x-cen)/wid))+const
    
    def double_sigmoid(self, x, amp1, cen1, wid1, amp2, cen2, wid2, const):
        return self.sigmoid(x, amp1, cen1, wid1, 0) + self.sigmoid(x, amp2, cen2, wid2, 0) + const

    def auto_borders(self):
        smooth_y = savgol_filter(self.y, 400, 2)    
        derivative = np.gradient(smooth_y)
        derivative = savgol_filter(derivative, 400, 2)
        derivative = np.abs(derivative)
        threshold = 0.1 * np.max(derivative)

        time = np.array(self.x)
        time_points = [time[i] for i in range(len(derivative)) if derivative[i] > threshold]
        
        if time_points == []:
            return -1, 0    
        left = time_points[0]
        right = time_points[-1]

        filtered_indices = np.where((time >= left) & (time <= right))[0]

        print(f"left: {left}, right: {right}")
        self.x = np.array(self.x)[filtered_indices]
        self.y = np.array(self.y)[filtered_indices]

    def fit(self, func):
        popt = super().fit(func, p0=[abs(np.max(self.y)-np.min(self.y)), 0.5*(self.x[0]+self.x[-1]), 0.05, -7],
                           bounds=([0, self.x[0], 0.000001, -np.inf], [np.inf, self.x[-1], np.inf, np.inf]))
        return np.append(popt, self.sign)

class TriggerFitter(Fitter):
    def __init__(self, x, y):
        super().__init__(x, y)

    def gaussian(self, x, amp, cen, wid, bg):
        return amp * np.exp(-(x-cen)**2 / 2/wid/wid)+bg
    
    def gauss(self, x, amp, cen, wid):
        return amp * np.exp(-(x-cen)**2 / 2/wid/wid)
