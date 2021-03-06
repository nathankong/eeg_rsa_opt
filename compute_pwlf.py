import pwlf
import numpy as np

def do_pwlf(data, start, end):
    # data: (num_timepoints,)
    # Fits a two segment piecewise linear equation and returns onset time

    # x_vals is hardcoded to the timespan we use
    x_vals = np.linspace(-112+(start*16),-112+(end*16),end-start+1)
    assert data.shape[0] == x_vals.shape[0]
    assert data.ndim == 1 and x_vals.ndim == 1

    p = pwlf.PiecewiseLinFit(x_vals, data)
    res = p.fit(2) # Two line segments (baseline and rise)

    assert p.n_segments == 2
    assert p.fit_breaks.shape[0] == 3
    onset = p.fit_breaks[1]

    return onset

