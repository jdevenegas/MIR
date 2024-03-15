import numpy as np
from fastdtw import fastdtw
from scipy.interpolate import interp1d

def resample_to_match_length(input_array, target_length, method='linear'):
    """
    Resamples an input array along the time axis (axis 0) to match the target length.
    
    Parameters:
        input_array (ndarray): Input array of shape (N, D) where N is the original length.
        target_length (int): Target length to which the input array will be resampled.
    
    Returns:
        resampled_array (ndarray): Resampled array of shape (target_length, D).
    """
    original_length = input_array.shape[0]
    if original_length == target_length:
        return input_array
    
    # Create interpolation function for each dimension
    interpolators = [interp1d(np.arange(original_length), input_array[:, i], kind=method) for i in range(input_array.shape[1])]
    
    # Resample along the time axis
    resampled_time_axis = np.linspace(0, original_length - 1, target_length)
    resampled_array = np.column_stack([interpolator(resampled_time_axis) for interpolator in interpolators])
    
    return resampled_array

def zero_pad(x,y):
    len_x = len(x)
    len_y = len(y)
    if len_x > len_y:
        y = np.pad(y, (0,len(x)-len(y)), 'constant', constant_values=(0))
    elif len_x < len_y:
        x = np.pad(x, (0,len(y)-len(x)), 'constant', constant_values=(0))
    return (x, y)

def sort_arrays_by_len(x,y):
    if len(x) >= len(y):
        return x, y
    else:
        return y, x
# def sliding_diff(x,y):
#     x, y = sort_arrays_by_len(x,y)
#     max_lag = len(x)-len(y)+1
#     # print(max_lag)
#     len_y = len(y)
#     total_diff = 0
#     for lag in range(max_lag):
#         # print('lag %d'%lag)
#         diff = (x[lag:len_y+lag]-y)**2
#         total_diff += diff.mean()
#         # print(diff.mean())
#     return total_diff

def cut_ts_ends(x, k=3, begin=True, end=True):
    median = np.median(x, axis=0)
    std = np.std(x, axis=0)
    v_min = median-k*std
    trim_cond = x>=v_min

    if begin:
        #get first time where x >= min bd - defaults to first dim
        # t_begin = np.argmax(trim_cond, axis=0)[0]
        #if we want first time where all dimensions exceed their min bd
        t_begin = np.argmax(trim_cond, axis=0).max()
    else:
        t_begin=0
    if end:
        #get final time where x >= min bd - defaults to first dim
        # t_end = len(x) - 1 - np.argmax(trim_cond[::-1], axis=0)[0]
        #if we want final time where all dimensions exceed their min bd
        t_end = len(x) - 1 - np.argmax(trim_cond[::-1], axis=0).max()
    else:
        t_end = len(x)-1
    return x[t_begin:t_end+1], np.arange(t_begin, t_end+1)


def sliding_diff(x,y, trim=False, k=np.inf):
    x, y = sort_arrays_by_len(x,y)
    total_diff = 0
    if trim:
        x, tx = cut_ts_ends(x, k=np.inf, begin=False, end=False)
        y_begin, ty_begin = cut_ts_ends(y, k=k, begin=False, end=True)
        y_middle, ty_middle = cut_ts_ends(y, k=k, begin=True, end=True)
        y_end, ty_end = cut_ts_ends(y, k=k, begin=True, end=False)
        max_lag = len(tx)-len(ty_end)+1

        for lag in range(max_lag):
            if lag==0:
                # print('begin')
                # pltx2, t2 = cut_ts_ends(x2, k=k, begin=False, end=True)
                y_cut, _ = y_begin, ty_begin
            elif lag==max_lag-1:
                # print('end')
                # pltx2, t2 = cut_ts_ends(x2, k=k, begin=True, end=False)
                y_cut, _ = y_end, ty_end
            else:
                # print('middle')
                # pltx2, t2 = cut_ts_ends(x2, k=k, begin=True, end=True)
                y_cut, _ = y_middle, ty_middle
            diff = (x[lag:len(y_cut)+lag]-y)**2
            total_diff += diff.mean()
    else:
        max_lag = len(x)-len(y)+1
        # print(max_lag)
        len_y = len(y)
        for lag in range(max_lag):
            # print('lag %d'%lag)
            diff = (x[lag:len_y+lag]-y)**2
            total_diff += diff.mean()
            # print(diff.mean())
    return total_diff
    
# def scale_minmax(X, minmax_scale = False, scale_min=0, scale_max=1, a=-1, b=1):
    
def pairwise_sliding_diffs(X, trim=False, k=np.inf, minmax_scale = False, scale_min=None, scale_max=None, a=-1, b=1):
    D = np.ones((len(X),len(X))) * np.inf

    if minmax_scale:
        if scale_min is None:
            scale_min = np.concatenate(X,axis=0).min(0)
        if scale_max is None:
            scale_max = np.concatenate(X,axis=0).max(0)
            
    for i in range(len(X)):
        x = X[i]
        if minmax_scale:
            x = scale_minmax(x, scale_min, scale_max, a, b)
        for j in range(i+1,len(X)):
            y = X[j]
            if minmax_scale:
                y = scale_minmax(y, scale_min, scale_max, a, b)
            # print(i,j)
            D[i,j] = sliding_diff(x,y, trim=trim, k=k)
    return D


def pairwise_diffs(X, resample=True, interpol_method = 'nearest', minmax_scale = False, scale_min=None, scale_max=None, a=-1, b=1):
    D = np.ones((len(X),len(X))) * np.inf
    
    if minmax_scale:
        if scale_min is None:
            scale_min = np.concatenate(X,axis=0).min(0)
        if scale_max is None:
            scale_max = np.concatenate(X,axis=0).max(0)
            
    for i in range(len(X)):
        x = X[i]
        if minmax_scale:
            x = scale_minmax(x, scale_min, scale_max, a, b)
        len_x = len(x)
        for j in range(i+1,len(X)):
            y = X[j]
            if minmax_scale:
                y = scale_minmax(y, scale_min, scale_max, a, b)
            if resample & (len(y) != len_x):
                if len(x)>len(y):
                    y_resampled = resample_to_match_length(y, len_x, method = interpol_method)
                    diff = (x-y_resampled)**2
                    D[i,j] = diff.mean()
                elif len(x)<len(y):
                    x_resampled = resample_to_match_length(x, len(y), method = interpol_method)
                    diff = (y-x_resampled)**2
                    D[i,j] = diff.mean()
            else:
                diff = (y-x)**2
                D[i,j] = diff.mean()
                
    return D


def scale_minmax(x, scale_min=None, scale_max=None, a=-1, b=1):
    if scale_min is None:
        scale_min = np.min(x, axis=0)
    if scale_max is None:
        scale_max = np.max(x, axis=0)

    x_scaled = a+(b-a)*(x-scale_min)/(scale_max-scale_min)
    return x_scaled
    

def pairwise_fastdtw(X, radius=1, dist=1, resample=True, interpol_method = 'nearest', 
                     minmax_scale=False, scale_min=None, scale_max=None, a=-1, b=1):
    
    D = np.ones((len(X),len(X))) * np.inf

    if minmax_scale:
        if scale_min is None:
            scale_min = np.concatenate(X,axis=0).min(0)
        if scale_max is None:
            scale_max = np.concatenate(X,axis=0).max(0)
            
    for i in range(len(X)):
        x = X[i]
        if minmax_scale:
            x = scale_minmax(x, scale_min, scale_max, a, b)
        len_x = len(x)
        for j in range(i+1,len(X)):
            y = X[j]
            if minmax_scale:
                y = scale_minmax(y, scale_min, scale_max, a, b)
            if resample & (len(y) != len_x):
                # print('resampling')
                if len(x)>len(y):
                    y_resampled = resample_to_match_length(y, len(x), method=interpol_method)
                    distance, _ = fastdtw(x, y_resampled, radius=radius, dist=dist)
                elif len(x)<len(y):
                    x_resampled = resample_to_match_length(x, len(y), method=interpol_method)
                    distance, _ = fastdtw(x_resampled, y, radius=radius, dist=dist)
            else:
                distance, _ = fastdtw(x, y, radius=radius, dist=dist)
                
            D[i,j] = distance
    return D


def sliding_fastdtw(x,y, trim=False, k=np.inf, radius=1, dist=1):
    x, y = sort_arrays_by_len(x,y)
    total_diff = 0
    if trim:
        x, tx = cut_ts_ends(x, k=np.inf, begin=False, end=False)
        y_begin, ty_begin = cut_ts_ends(y, k=k, begin=False, end=True)
        y_middle, ty_middle = cut_ts_ends(y, k=k, begin=True, end=True)
        y_end, ty_end = cut_ts_ends(y, k=k, begin=True, end=False)
        max_lag = len(tx)-len(ty_end)+1

        for lag in range(max_lag):
            if lag==0:
                # print('begin')
                # pltx2, t2 = cut_ts_ends(x2, k=k, begin=False, end=True)
                y_cut, _ = y_begin, ty_begin
            elif lag==max_lag-1:
                # print('end')
                # pltx2, t2 = cut_ts_ends(x2, k=k, begin=True, end=False)
                y_cut, _ = y_end, ty_end
            else:
                # print('middle')
                # pltx2, t2 = cut_ts_ends(x2, k=k, begin=True, end=True)
                y_cut, _ = y_middle, ty_middle
            diff, _ = fastdtw(x, y, radius=radius, dist=dist)
            total_diff += diff #.mean()
    else:
        max_lag = len(x)-len(y)+1
        # print(max_lag)
        len_y = len(y)
        for lag in range(max_lag):
            # print('lag %d'%lag)
            diff, _ = fastdtw(x, y, radius=radius, dist=dist)
            total_diff += diff #.mean()
            # print(diff.mean())
    return total_diff


def pairwise_sliding_fastdtw(X, trim=False, k=np.inf, radius=1, dist=1, minmax_scale=False, scale_min=None, scale_max=None, a=-1, b=1):
    D = np.ones((len(X),len(X))) * np.inf

    if minmax_scale:
        if scale_min is None:
            scale_min = np.concatenate(X,axis=0).min(0)
        if scale_max is None:
            scale_max = np.concatenate(X,axis=0).max(0)
            
    for i in range(len(X)):
        x = X[i]
        if minmax_scale:
            x = scale_minmax(x, scale_min, scale_max, a, b)
        for j in range(i+1,len(X)):
            y = X[j]
            if minmax_scale:
                y = scale_minmax(y, scale_min, scale_max, a, b)
            D[i,j] = sliding_fastdtw(x,y, trim=trim, k=np.inf, radius=radius, dist=dist)
    return D