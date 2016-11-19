import numpy as np

def avg_hem(lat, data, neg=False):
    """Averages data between the NH & SH
       inputs---
         lat:  1-d array of global latitudes. Should be even
         data: 1-d or 2-d array of data with latitude dimension first
         neg : Boolean of whether to do asymmetric hemisphere avg
       outputs---
         lat: NH lat
         out: Hemispherically-averaged data"""
    if min(lat) >= 0.:
        raise ValueError('input lat should be SH & NH!')
    if np.mod(np.size(lat)/2, 1) != 0.0:
        raise ValueError('Expecting even number of latitudes')
    if np.size(lat) != np.shape(data)[0]:
        raise ValueError('Latitude must be on the first dimension')
    N = int(np.size(lat)/2)
    if np.ndim(data) == 1:
        data = data[:,None]
    elif np.ndim(data) > 2:
        raise ValueError('Data must be 1-d or 2-d only');
    z1 = data[:N,:,None][::-1]
    z2 = data[N:,:,None]
    if neg:
        z = np.concatenate((-z1, z2), axis=2)
    else:
        z = np.concatenate((z1, z2), axis=2)
    out = np.squeeze(np.mean(z, axis=2))
    lat = lat[N:]
    return lat,out

def load_serial(exper,var):
    """Averages data from all serial files"""
    ptf = '/disk7/jgdwyer/chickpea/idealized_runs/'
#def three_panel_compare_climo():
