import numpy as np

def add_new_data(data: np.ndarray, new_data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return new_data
    return np.concatenate((data, new_data))


def generate_window(window_type: str, window_len: int, num_rows: int, normalize: bool) -> np.ndarray:
    if window_type == "flat":
        window = np.ones(window_len)
    elif window_type == "hanning":
        window = np.hanning(window_len)
    elif window_type == "hamming":
        window = np.hamming(window_len)
    elif window_type == "bartlett":
        window = np.bartlett(window_len)
    elif window_type == "blackman":
        window = np.blackman(window_len)
    else:
        raise ValueError(f"Unsupported window type '{window_type}'.")

    if normalize:
        window /= window.sum()

    return window  # Always return 1D


# def incremental_convolve(array_slice: np.ndarray, window: np.ndarray) -> float:
#     if array_slice.size != window.size:
#         raise ValueError("Input and kernel must be the same size for real-time convolution.")
#     return float(np.dot(array_slice[-window.size:], window))

def incremental_convolve(data, window_len, window):
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        
        y = np.convolve(w / w.sum(), data, mode='valid')
        return y[0]


def incremental_gradient(dy: np.ndarray, dx: np.ndarray) -> float:
    update_dy = dy[-3:]
    update_dx = dx[-3:]

    grad_dx = np.gradient(update_dx)
    if np.any(np.abs(grad_dx) < 1e-3):
        custom_dx = np.array([0.001, 0.002, 0.003])
        grad = np.gradient(update_dy, custom_dx)
    else:
        grad = np.gradient(update_dy, update_dx)

    return float(grad[-1])


# def smooth_alternative(array: np.ndarray, window: np.ndarray) -> np.ndarray:
#     pad_width = len(window) // 2
#     padded = np.pad(array, pad_width, mode='reflect')
#     return np.convolve(padded, window, mode='valid')


def smooth_alternative(x, window_len=100, window='blackman', compensate_offset=True):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # expansion for convolution later
    # translation = int((window_len - 1)/2)
    s = np.r_[x[window_len - 1:0:-1], x]
    # s = np.r_[x[translation:0:-1], x, x[-1:-translation-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    
    
    return y

def slope(x: np.ndarray, range_: int) -> float:
    if x.size < range_:
        raise ValueError("x size is smaller than range")
    x_range = x[-range_:]
    return (x_range[-1] - x_range[0]) / (range_ - 1)
