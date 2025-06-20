import numpy as np
from threading import Lock
from fixing_utils import (
    generate_window,
    smooth_alternative,
    incremental_convolve,
    incremental_gradient,
    add_new_data,
    slope,
)


class FFAnalyzerRTSlide:
    def __init__(self):
        self.find_contactlos = False
        self.stdev_converge = False
        self.data_size = 0
        self.ending_point = 0
        self.time_step = 0.001

        self.crt_vector = []
        self.mean_vector = []
        self.std_vector = []

        self.crt_vector_mutex = Lock()

    def set_window_size(self, window_size):
        self.window_len = window_size
        self.mvg_window = generate_window("flat", window_size, 1, True)
        self.bartlett_window = generate_window("bartlett", window_size, 1, True)

        self.buffer_len = window_size
        self.fext_y_buffer = np.zeros(self.buffer_len)
        self.ff_y_buffer = np.zeros(self.buffer_len)
        self.ts_buffer = np.zeros(self.buffer_len)

    def set_crt_type(self, type_name):
        self.crt_type = type_name

    def set_stat_type(self, type_name):
        self.stat_type = type_name

    def set_z_score(self, val):
        self.z_score = val

    def fifo_append(self, buffer, new_value):
        buffer[:-1] = buffer[1:]
        buffer[-1] = new_value
        return buffer

    def update_sensor_data(self, new_fext_value, new_ff_value):
        if self.data_size < self.buffer_len:
            self.fext_y_buffer[self.data_size] = new_fext_value
            self.ff_y_buffer[self.data_size] = new_ff_value
        else:
            self.fifo_append(self.fext_y_buffer, new_fext_value)
            self.fifo_append(self.ff_y_buffer, new_ff_value)

        self.data_size += 1
        ts_series = np.arange(self.data_size)
        self.ts_buffer = self.time_step * ts_series

        if self.data_size == self.window_len:
            self.init()
        elif self.data_size > self.window_len:
            self.real_time_detection(self.z_score)

    def init(self):
        self.fext_y_smt = smooth_alternative(self.fext_y_buffer, self.mvg_window)
        self.crt_vector = [[0.0] for _ in range(self.window_len)]
        self.dfext_dff_smt_vector = [[0.0] for _ in range(self.window_len)]

        if self.crt_type == "dep":
            self.ts = 0.001 * np.arange(self.data_size)
            self.dfext_dff = incremental_gradient(self.fext_y_smt, self.ts)
            self.dfext_dff_smt = smooth_alternative(self.dfext_dff, self.bartlett_window)
            self.mean_vector.append([np.mean(self.dfext_dff_smt)])
            self.std_vector.append([np.std(self.dfext_dff_smt)])

    def real_time_detection(self, threshold):
        f = self.data_size - 1
        new_smt_fext = incremental_convolve(self.fext_y_buffer, self.mvg_window)
        self.fifo_append(self.fext_y_smt, new_smt_fext)

        if self.crt_type == "dep":
            new_dfext_dff = incremental_gradient(self.fext_y_smt, self.ff_y_buffer)

            if not hasattr(self, "dfext_dff"):
                self.dfext_dff = np.zeros(self.buffer_len)
            self.fifo_append(self.dfext_dff, new_dfext_dff)

            new_smt_dfext_dff = incremental_convolve(self.dfext_dff, self.bartlett_window)

            if not hasattr(self, "dfext_dff_smt"):
                self.dfext_dff_smt = np.array([])

            if self.stat_type == "cumul_zscore":
                self.dfext_dff_smt = add_new_data(self.dfext_dff_smt, np.array([new_smt_dfext_dff]))
            elif self.stat_type in ["roll_zscore", "roll_slope"]:
                if self.dfext_dff_smt.size < self.buffer_len:
                    self.dfext_dff_smt = add_new_data(self.dfext_dff_smt, np.array([new_smt_dfext_dff]))
                else:
                    self.fifo_append(self.dfext_dff_smt, new_smt_dfext_dff)

            avg = np.mean(self.dfext_dff_smt)
            std = np.std(self.dfext_dff_smt)
            self.mean_vector.append([avg])
            self.std_vector.append([std])

            if f > 2 * self.window_len and std < 1.5:
                if not self.stdev_converge:
                    print(f"converge at: {f}")
                self.stdev_converge = True

            if self.stdev_converge:
                if self.stat_type == "cumul_zscore":
                    criterion = (new_smt_dfext_dff - avg) / std
                    print(f"cumulative criterion: {criterion}")
                elif self.stat_type == "roll_zscore":
                    if not hasattr(self, "slope_buffer"):
                        self.slope_buffer = np.array([])
                    new_slope = slope(self.dfext_dff_smt, 10)
                    self.slope_buffer = add_new_data(self.slope_buffer, np.array([new_slope]))
                    slope_mean = np.mean(self.slope_buffer)
                    slope_std = np.std(self.slope_buffer)
                    criterion = (new_slope - slope_mean) / slope_std
                elif self.stat_type == "roll_slope":
                    criterion = slope(self.dfext_dff_smt, 10)

                with self.crt_vector_mutex:
                    self.crt_vector.append([criterion])
                if criterion < -threshold:
                    self.find_contactlos = True
                    self.ending_point = f - 1
                    print(f"lose contact at {self.ending_point}")
            else:
                self.crt_vector.append([0.0])
            self.dfext_dff_smt_vector.append([new_smt_dfext_dff])

    def get_window_size(self):
        return self.window_len

    def get_contactlos(self):
        return False if self.crt_type == "none" else self.find_contactlos

    def get_data_size(self):
        return self.data_size

    def get_ending_point(self):
        return self.ending_point

    def get_dfext_dff_smt(self):
        return self.dfext_dff_smt_vector

    def get_mean(self):
        return self.mean_vector

    def get_std(self):
        return self.std_vector

    def get_criterion(self):
        with self.crt_vector_mutex:
            return self.crt_vector
