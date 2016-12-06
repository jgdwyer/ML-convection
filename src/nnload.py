import numpy as np
from netCDF4 import Dataset
from sklearn import preprocessing, metrics
import scipy.stats
import pickle
import warnings


def loaddata(filename, minlev, all_lats=True, indlat=None, N_trn_exs=None,
             rainonly=False, noshallow=False, cosflag=True, randseed=False,
             verbose=True):
    """v2 of the script to load data. See prep_convection_output.py for how
       the input filename is generated.

    Args:
      filename:  The file to be loaded. Use './data/convcond_training_v3.pkl'
                 or './data/convcond_testing_v3.pkl'
      minlev:    The topmost model level for which to load data. Set to 0. to
                 load all data
      all_lats:  Logical value for whether to load data from all latitudes
      indlat:    If all_lats is false, give the index value [0-31] for the
                 latitude at which to load data.
      N_trn_exs: Number of training examples to load. If set to None, or
                 if requested number exceeds max available will load all.
      rainonly:  If true, only return training examples of when it is raining
      noshallow: If true, only return training examples of when the shallow
                 convection scheme does NOT happen. (So, only return examples
                 with deep convection, or no convection at all)
      cosflag:   If true, use cos(lat) weighting for loading training examples
      randseed:  If true, seed the random generator to a recreateable state
      verbose:   If true, prints some basic stats about training set

    Returns:
      x       : 2-d numpy array of input features (m_training examples x
                N_input features). If minlev is 0., there will be 60 input
                features, the top 30 for temperature and the bottom 30 for
                humidity.
      y       : 2-d numpy array of output targets (m_traning examples x
                N_output targets). If minlev is 0., there will be 60 output
                features, the top 30 for temp. tendencies and the bottom 30
                for q tend.
      cv      : 1-d array (m_training examples x 1) that gives 1 if convection
                occurs and 0 if it does not.
      Pout    : 1-d arrray (m_training examples x 1) of how much precipitation
                occurs in kg/m^2/s (multiply by 3600*24 to convert
                precipitation to mm/day)
      lat2    : 1-d array of latitude for one hemisphere (since hemispheres
                are combined)
      lev     : The vertical model levels (1 is the surface and 0 is the top
                of the atmosphere).
      dlev    : The difference between model levels, useful for calculating
                some derived quantities.
      timestep: How large each model timestep is in seconds.
    """
    # Data to read in is N_lev x N_lat (SH & NH) x N_samples
    # Samples are quasi indpendent with only 5 from each latitude range chosen
    # randomly over different longitudes and times within that 24 hour period.
    # Need to use encoding because saved using python2:
    # http://stackoverflow.com/q/28218466
    v = dict()
    [v['Tin'], v['qin'], v['Tout'], v['qout'], Pout, lat] = \
        pickle.load(open(filename, 'rb'), encoding='latin1')
    # Use this to calculate the real sigma levels
    lev, dlev, indlev = get_levs(minlev)
    varis = ['Tin', 'qin', 'Tout', 'qout']
    # Reshape the arrays
    for var in varis:
        # Change shape of data to be N_samp x N_lev
        if all_lats:
            # print('error')
            if cosflag:
                v[var] = reshape_cos_lats(v[var], indlev, lat)
            else:
                v[var] = reshape_all_lats(v[var], indlev)
        else:
            if indlat is not None:
                v[var] = reshape_one_lat(v[var], indlev, indlat)
            else:
                raise TypeError('Need to set an index value for indlat')
    # Also reshape precipitation
    if all_lats:
        if cosflag:
            Pout = reshape_cos_lats(Pout, None, lat, is_precip=True)
        else:
            # Need to do a transpose to be consistent with reshape_all_lats
            Pout = np.reshape(Pout.transpose(), -1)
    else:
        Pout = Pout[indlat, :]
    # Randomize the order of these events
    m = v['Tin'].shape[0]
    if randseed:
        np.random.seed(0)
    randind = np.random.permutation(m)
    for var in varis:
        v[var] = v[var][randind, :]
    Pout = Pout[randind]
    # Converted heating rates to K/day and g/kg/day in
    # prep_convection_output.py
    # Concatenate input and output variables together
    x = pack(v['Tin'], v['qin'], axis=1)
    y = pack(v['Tout'], v['qout'], axis=1)
    # The outputs get lined up in prep_convection_output.py
    # Print some statistics about rain and limit to when it's raining if True
    x, y, Pout = limitrain(x, y, Pout, rainonly, noshallow=noshallow,
                           verbose=verbose)
    # Limit to only certain events if requested
    if N_trn_exs is not None:
        if N_trn_exs > y.shape[0]:
            warnings.warn('Requested more samples than available. Using the' +
                          'maximum number available')
            N_trn_exs = y.shape[0]
        ind = np.arange(N_trn_exs)
        x = x[ind, :]
        y = y[ind, :]
        Pout = Pout[ind]
    # Store when convection occurs
    cv, _ = whenconvection(y, verbose=verbose)
    timestep = 10*60  # 10 minute timestep in seconds
    return x, y, cv, Pout, lat, lev, dlev, timestep


def reshape_cos_lats(z, indlev, lat, is_precip=False):
    if is_precip:
        z = z.swapaxes(0, 1)
        z2 = np.empty((0))
    else:
        z = z[indlev, :, :]
        z = z.swapaxes(0, 2)
        z2 = np.empty((0, sum(indlev)))
    N_ex = z.shape[0]
    for i, latval in enumerate(lat):
        Ninds = int(N_ex * np.cos(np.deg2rad(latval)))
        if is_precip:
            z2 = np.concatenate((z2, z[0: Ninds, i]), axis=0)
        else:
            z2 = np.concatenate((z2, z[0:Ninds, i, :]), axis=0)
    return z2


def reshape_all_lats(z, indlev):
    # Expects data to be N_lev x N_lat x N_samples and returns
    # (N_lat*N_samp x N_lev)
    z = z[indlev, :, :]
    z = z.swapaxes(0, 2)
    return np.reshape(z, (-1, sum(indlev)))


def reshape_one_lat(z, indlev, indlat):
    # Expects data to be N_lev x N_lat x N_samples and returns (N_samp x N_lev)
    z = z[indlev, indlat, :]
    z = z.swapaxes(0, 1)
    return z


def pack(d1, d2, axis=1):
    """Combines T & q profiles as an input matrix to NN"""
    return np.concatenate((d1, d2), axis=axis)


def unpack(data, vari, axis=1):
    """Reverse pack operation to turn ouput matrix into T & q"""
    N = int(data.shape[axis]/2)
    varipos = {'T': np.arange(N), 'q': np.arange(N, 2*N)}
    out = np.take(data, varipos[vari], axis=axis)
    return out


# Initialize & fit scaler
def init_pp(ppi, raw_data):
    # Initialize list of scaler objects
    if ppi['name'] == 'MinMax':
        pp = [preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0)),  # temp
              preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))]  # humid.
    elif ppi['name'] == 'MaxAbs':
        pp = [preprocessing.MaxAbsScaler(),  # for temperature
              preprocessing.MaxAbsScaler()]  # and humidity
    elif ppi['name'] == 'StandardScaler':
        pp = [preprocessing.StandardScaler(),  # for temperature
              preprocessing.StandardScaler()]  # and humidity
    elif ppi['name'] == 'RobustScaler':
        pp = [preprocessing.RobustScaler(),  # for temperature
              preprocessing.RobustScaler()]  # and humidity
    elif ppi['name'] == 'SimpleY':
        pp = [10./1., 10./2.5]  # for temperature
    else:
        ValueError('Incorrect scaler name')
    # Initialize scalers with data
    if ppi['method'] == 'individually':
        pp[0].fit(unpack(raw_data, 'T'))
        pp[1].fit(unpack(raw_data, 'q'))
    elif ppi['method'] == 'alltogether':
        pp[0].fit(np.reshape(unpack(raw_data, 'T'), (-1, 1)))
        pp[1].fit(np.reshape(unpack(raw_data, 'q'), (-1, 1)))
    elif ppi['method'] == 'qTindividually':
        if ppi['name'] != 'SimpleY':
            pp = pp[0]
            pp.fit(raw_data)
    else:
        raise ValueError('Incorrect scaler method')
    return pp


# Transform data using initialized scaler
def transform_data(ppi, pp, raw_data):
    if ppi['method'] == 'individually':
        T_data = pp[0].transform(unpack(raw_data, 'T'))
        q_data = pp[1].transform(unpack(raw_data, 'q'))
    elif ppi['method'] == 'alltogether':
        T_data = pp[0].transform(np.reshape(unpack(raw_data, 'T'), (-1, 1)))
        q_data = pp[1].transform(np.reshape(unpack(raw_data, 'q'), (-1, 1)))
        # Return to original shape (N_samples x N_features) rather than
        # (N_s*N_f x 1)
        shp = unpack(raw_data, 'T').shape
        T_data = np.reshape(T_data, shp)
        q_data = np.reshape(q_data, shp)
    elif ppi['method'] == 'qTindividually':
        if ppi['name'] == 'SimpleY':
            T_data = unpack(raw_data, 'T')/pp[0]
            q_data = unpack(raw_data, 'q')/pp[1]
        else:
            all_data = pp.transform(raw_data)
            T_data = unpack(all_data, 'T')
            q_data = unpack(all_data, 'q')
    else:
        print('Given method is ' + ppi['method'])
        raise ValueError('Incorrect scaler method')
    # Return single transformed array as output
    return pack(T_data, q_data)


# Apply inverse transformation to unscale data
def inverse_transform_data(ppi, pp, trans_data):
    if ppi['method'] == 'individually':
        T_data = pp[0].inverse_transform(unpack(trans_data, 'T'))
        q_data = pp[1].inverse_transform(unpack(trans_data, 'q'))
    elif ppi['method'] == 'alltogether':
        T_data = pp[0].inverse_transform(np.reshape(unpack(trans_data, 'T'),
                                                    (-1, 1)))
        q_data = pp[1].inverse_transform(np.reshape(unpack(trans_data, 'q'),
                                                    (-1, 1)))
        # Return to original shape (N_samples x N_features) rather than
        # (N_s*N_f x 1)
        shp = unpack(trans_data, 'T').shape
        T_data = np.reshape(T_data, shp)
        q_data = np.reshape(q_data, shp)
    elif ppi['method'] == 'qTindividually':
        if ppi['name'] == 'SimpleY':
            T_data = unpack(trans_data, 'T') * pp[0]
            q_data = unpack(trans_data, 'q') * pp[1]
        else:
            all_data = pp.inverse_transform(trans_data)
            T_data = unpack(all_data, 'T')
            q_data = unpack(all_data, 'q')
    else:
        raise ValueError('Incorrect scaler method')
    # Return single transformed array as output
    return pack(T_data, q_data)


def subsample(x, y, N_samples=0):
    """Preprocess data by splitting it into 3 equally sized samples"""
    if N_samples == 0:
        N_samples = x.shape[0]
    # Randomly choose samples
    samples = np.random.choice(x.shape[0], N_samples, replace=False)
    # Split data
    ss = int(np.floor(len(samples)/3))  # number of samples in each set

    def split_sample(z, samples, ss):
        z1 = np.take(z, samples[0:ss], axis=0)
        z2 = np.take(z, samples[ss:2*ss], axis=0)
        z3 = np.take(z, samples[2*ss:3*ss], axis=0)
        return z1, z2, z3
    # Apply to input and output data
    x1, x2, x3 = split_sample(x, samples, ss)
    y1, y2, y3 = split_sample(y, samples, ss)
    return x1, x2, x3, y1, y2, y3


def limitrain(x, y, Pout, rainonly=False, noshallow=False, verbose=True):
    indrain = np.greater(Pout, 0)
    if verbose:
        print('There is some amount of rain {:.1f}% of the time'.
              format(100. * np.sum(indrain)/len(indrain)))
        print('There is a rate of >3 mm/day {:.1f}% of the time'.
              format(100. * np.sum(np.greater(Pout, 3))/len(indrain)))
    if rainonly:
        x = x[indrain, :]
        y = y[indrain, :]
        Pout = Pout[indrain]
        if verbose:
            print('Only looking at times it is raining!')
    if noshallow:
        cv, _ = whenconvection(y, verbose=True)
        indnosha = np.logical_or(Pout > 0, cv == 0)
        x = x[indnosha, :]
        y = y[indnosha, :]
        Pout = Pout[indnosha]
        if verbose:
            print('Excluding all shallow convective events...')
    return x, y, Pout


def whenconvection(y, verbose=True):
    """Caluclate how often convection occurs...useful for classification
       Also store a variable that is 1 if convection and 0 if no convection"""
    cv_strength = np.sum(np.abs(unpack(y, 'T')), axis=1)
    cv = np.copy(cv_strength)
    cv[cv > 0] = 1
    if verbose:
        print('There is convection {:.1f}% of the time'.
              format(100. * np.sum(cv)/len(cv)))
    return cv, cv_strength


def write_netcdf_onelayer(mlp, mlp_str, x_pp, y_pp,
                          filename='/Users/jgdwyer/neural_weights_v3.nc'):
    # Grab weights and input normalization
    w1 = mlp.get_parameters()[0].weights
    w2 = mlp.get_parameters()[1].weights
    b1 = mlp.get_parameters()[0].biases
    b2 = mlp.get_parameters()[1].biases
    xscale_mean = x_pp.mean_
    xscale_stnd = x_pp.scale_
    Nlev = int(np.shape(b2)[0]/2)
    yscale_absmax = np.zeros(b2.shape)
    yscale_absmax[:Nlev] = y_pp[0]
    yscale_absmax[Nlev:] = y_pp[1]
    # Write weights to file
    ncfile = Dataset(filename, 'w')
    # Write the dimensions
    ncfile.createDimension('N_in',     w1.shape[0])
    ncfile.createDimension('N_h1',     w1.shape[1])
    ncfile.createDimension('N_out',    w2.shape[1])
    # Create variable entries in the file
    nc_w1 = ncfile.createVariable('w1', np.dtype('float64').char,
                                  ('N_h1', 'N_in'))  # Reverse dims
    nc_w2 = ncfile.createVariable('w2', np.dtype('float64').char,
                                  ('N_out', 'N_h1'))
    nc_b1 = ncfile.createVariable('b1', np.dtype('float64').char,
                                  ('N_h1'))
    nc_b2 = ncfile.createVariable('b2', np.dtype('float64').char,
                                  ('N_out'))
    nc_xscale_mean = ncfile.createVariable('xscale_mean',
                                           np.dtype('float64').char, ('N_in'))
    nc_xscale_stnd = ncfile.createVariable('xscale_stnd',
                                           np.dtype('float64').char, ('N_in'))
    nc_yscale_absmax = ncfile.createVariable('yscale_absmax',
                                             np.dtype('float64').char,
                                             ('N_out'))
    # Write variables and close file - transpose because fortran reads it in
    # "backwards"
    nc_w1[:] = w1.T
    nc_w2[:] = w2.T
    nc_b1[:] = b1
    nc_b2[:] = b2
    nc_xscale_mean[:] = xscale_mean
    nc_xscale_stnd[:] = xscale_stnd
    nc_yscale_absmax[:] = yscale_absmax
    # Write global file attributes
    ncfile.description = mlp_str
    ncfile.close()


def avg_hem(data, lat, axis, split=False):
    """Averages the NH and SH data (or splits them into two data sets)"""
    ixsh = np.where(lat < 0)[0]  # where returns a tuple
    ixnh = np.where(lat >= 0)[0]
    if len(ixsh) == 0:
        print(lat)
        raise ValueError('Appears that lat does not have SH values')
    lathalf = lat[ixnh]
    sh = np.take(data, ixsh, axis=axis)
    nh = np.take(data, ixnh, axis=axis)
    # Flip the direction of the sh data at a given axis
    shrev = np.swapaxes(np.swapaxes(sh, 0, axis)[::-1], 0, axis)
    # If splitting data, return these arrays
    if split:
        return nh, shrev, lathalf
    else:
        return (nh + shrev) / 2., lathalf


def load_one_lat(x_ppi, y_ppi, x_pp, y_pp, r_mlp, indlat, datafile, minlev=0.,
                 rainonly=False):
    """Returns N_samples x 2*N_lev array of true and predicted values
       at a given latitude"""
    # Load data
    x, y, cv, Pout, lat, lev, dlev, timestep = \
        loaddata(datafile, minlev, rainonly=rainonly, all_lats=False,
                 indlat=indlat, verbose=False, N_trn_exs=2500)
    # Calculate predicted output
    x = transform_data(x_ppi, x_pp, x)
    y_pred = r_mlp.predict(x)
    y_pred = inverse_transform_data(y_ppi, y_pp, y_pred)
    # Output true and predicted temperature and humidity tendencies
    T = unpack(y, 'T')
    q = unpack(y, 'q')
    T_pred = unpack(y_pred, 'T')
    q_pred = unpack(y_pred, 'q')
    return T, q, T_pred, q_pred


def stats_by_latlev(x_ppi, y_ppi, x_pp, y_pp, r_mlp, lat, lev, datafile):
    # Initialize
    Tmean = np.zeros((len(lat), len(lev)))
    qmean = np.zeros((len(lat), len(lev)))
    Tbias = np.zeros((len(lat), len(lev)))
    qbias = np.zeros((len(lat), len(lev)))
    rmseT = np.zeros((len(lat), len(lev)))
    rmseq = np.zeros((len(lat), len(lev)))
    rT = np.zeros((len(lat), len(lev)))
    rq = np.zeros((len(lat), len(lev)))
    for i in range(len(lat)):
        print(i)
        T_true, q_true, T_pred, q_pred = \
            load_one_lat(x_ppi, y_ppi, x_pp, y_pp, r_mlp, i, datafile,
                         minlev=np.min(lev))
        # Get means of true output
        Tmean[i, :] = np.mean(T_true, axis=0)
        qmean[i, :] = np.mean(q_true, axis=0)
        # Get bias from means
        Tbias[i, :] = np.mean(T_pred, axis=0) - Tmean[i, :]
        qbias[i, :] = np.mean(q_pred, axis=0) - qmean[i, :]
        # Get rmse
        rmseT[i, :] = np.sqrt(
            metrics.mean_squared_error(T_true, T_pred,
                                       multioutput='raw_values'))
        rmseq[i, :] = np.sqrt(
            metrics.mean_squared_error(q_true, q_pred,
                                       multioutput='raw_values'))
        # Get correlation coefficients
        for j in range(len(lev)):
            rT[i, j], _ = scipy.stats.pearsonr(T_true[:, j], T_pred[:, j])
            rq[i, j], _ = scipy.stats.pearsonr(q_true[:, j], q_pred[:, j])
    return Tmean.T, qmean.T, Tbias.T, qbias.T, rmseT.T, rmseq.T, rT.T, rq.T


def get_levs(minlev):
    # Define half sigma levels for data
    half_lev = np.array([0.000000000000000e+00, 9.202000000000000e-03,
                         1.244200000000000e-02, 1.665600000000000e-02,
                         2.207400000000000e-02, 2.896500000000000e-02,
                         3.762800000000000e-02, 4.839600000000000e-02,
                         6.162600000000000e-02, 7.769200000000000e-02,
                         9.697200000000000e-02, 1.198320000000000e-01,
                         1.466070000000000e-01, 1.775800000000000e-01,
                         2.129570000000000e-01, 2.528400000000000e-01,
                         2.972050000000000e-01, 3.458790000000000e-01,
                         3.985190000000000e-01, 4.546020000000000e-01,
                         5.134170000000000e-01, 5.740720000000000e-01,
                         6.355060000000000e-01, 6.965140000000000e-01,
                         7.557840000000000e-01, 8.119360000000000e-01,
                         8.635820000000000e-01, 9.093730000000000e-01,
                         9.480640000000000e-01, 9.785660000000000e-01,
                         1.000000000000000e+00])
    # Calculate the full levels
    lev = np.array(np.zeros((half_lev.size-1,)))
    for i in range(half_lev.size-1):
        lev[i] = (half_lev[i] + half_lev[i+1])/2.
    # Limit levels to those specified
    indlev = np.greater_equal(lev, minlev)
    lev = lev[indlev]
    # Calculate the spacing between levels
    dlev = np.diff(half_lev)
    dlev = dlev[indlev]
    return lev, dlev, indlev


def get_pred_true_from_mlp(r_str, minlev):
    # Load model and preprocessors
    mlp, _, errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, _ = \
        pickle.load(open('./data/regressors/' + r_str + '.pkl', 'rb'))
    # Load raw data from file
    x_unscl, ytcv_unscl, _, _, _, _, _, _ = \
        loaddata('./data/conv_testing_v3.pkl', minlev=minlev,
                 N_trn_exs=None)
    # Scale true values
    ytrue_scl = transform_data(y_ppi, y_pp, ytcv_unscl)
    # Apply x preprocessing to scale x-data and predict output
    x_scl = transform_data(x_ppi, x_pp, x_unscl)
    ypred_scl = mlp.predict(x_scl)
    return ypred_scl, ytrue_scl


def calc_mse_simple(y_pred, y_true, relflag=False, minlev=None, lev=None):
    """Calculates the mean squared error for scaled outputs and is taken over
       all training examples and levels of T and q
       Args:
        y_pred (float: Nex x N_lev*2): Scaled predicted value of output
        y_true (float Nex x N_lev*2): Scaled true value of output
        relflag (bool): Divide by standard deviation for each var at each lev
                        to somewhat account for difference in strength between
                        convection-only and conv+cond predictions
        minlev (float): Don't consider data above this level when calculating
                        mse
        lev (float: N_lev): Used for calculating minlev
       Returns:
        mse (float, scalar): Mean-Squared error """
    # If requested cut off certain levels
    if minlev is not None:
        if lev is None:
            raise ValueError('Also need to input levels!')
        indlev = np.greater_equal(lev, minlev)
        # For predicted data
        Tp = unpack(y_pred, 'T')[:, indlev]
        qp = unpack(y_pred, 'q')[:, indlev]
        y_pred = pack(Tp, qp)
        # For true data
        Tt = unpack(y_true, 'T')[:, indlev]
        qt = unpack(y_true, 'q')[:, indlev]
        y_true = pack(Tt, qt)
    # Calculate squared difference
    mse = np.square(y_pred - y_true)
    # If looking at relative mean squared error divide by the mean over all
    # examples for each variable and at each level
    if relflag:
        mse = mse/np.abs(np.std(y_true, axis=1))[:, None]
    # Now take mean value over flattened array for all values
    mse = np.mean(mse[np.isfinite(mse)])
    return mse
