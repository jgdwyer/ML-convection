import numpy as np
from netCDF4 import Dataset
from sklearn import preprocessing, metrics
import scipy.stats

def loaddata(filename, minlev, all_lats=True, indlat=None, rainonly=False, verbose=True):
    f = Dataset(filename, mode='r')
    timestep = 10*60 # 10 minute timestep
    # Read the data
    Tin = f.variables['temp'][:] # N_time x N_lev x N_lat x N_lon
    qin = f.variables['sphum'][:] 
    Tout= f.variables['dt_tg_convection'][:]
    qout= f.variables['dt_qg_convection'][:]
    Pout= f.variables['convection_rain'][:] # N_time x N_lat x N_lon
    lat = f.variables['lat'][:]
    #lev = f.variables['pfull'][:] / 1000. # sigma units
    f.close()
    # Use this to calculate the real sigma levels
    half_lev=np.array([0.000000000000000e+00, 9.202000000000000e-03, 
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
    lev = np.array(np.zeros((half_lev.size-1,)))
    for i in range(half_lev.size-1):
        lev[i] = (half_lev[i] + half_lev[i+1])/2.
    # Limit levels to those specified
    indlev = np.greater(lev, minlev)
    lev=lev[indlev]
    # Calculate the distance between levels
    dlev = np.diff(half_lev)
    dlev = dlev[indlev]
    # Apply preprocessing to all data
    if all_lats:
        Tin =  prep_all_lats(Tin ,indlev)
        qin =  prep_all_lats(qin ,indlev)
        Tout = prep_all_lats(Tout,indlev)
        qout = prep_all_lats(qout,indlev)
        Pout = prep_all_lats(Pout,indlev)
    # Or data at just one latitude
    else:
        if indlat is not None:
            Tin =  prep(Tin ,indlev, indlat)
            qin =  prep(qin ,indlev, indlat)
            Tout = prep(Tout,indlev, indlat)
            qout = prep(qout,indlev, indlat)
            Pout = prep(Pout,indlev, indlat)
        else:
            raise TypeError('Need to set an index value for indlat')    
    # Convert heating rates to K/day and g/kg/day
    Tout = Tout*3600.*24.
    qout = qout*3600.*24.*1000.
    # Concatenate input and output variables together
    x = pack(Tin,  qin , axis=1)
    y = pack(Tout, qout, axis=1)
    # Ensure that input and outputs are lined up in time (and warn user)
    import warnings
    warnings.warn("Shifting inputs and outputs one time step so they line up!")
    x=x[1:,:]
    y=y[0:-1,:]
    # Print some statistics about rain and limit to when it's raining if True
    x, y, Pout = limitrain(x, y, Pout, rainonly, verbose=verbose)
    # Store when convection occurs
    cv = whenconvection(y, verbose=verbose)
    return (x, y, cv, Pout, lat, lev, dlev, timestep)

def prep(M,indlev,indlat):
    if M.ndim == 4:
        M = M[:,indlev,indlat,:].squeeze()
        M = M.swapaxes(1,2) # N_time x N_lon x N_lev 
        M = np.reshape(M, (-1, len(indlev))) #Now N_time*N_lon x N_lev
    elif M.ndim == 3:
        M = M[:,indlat,:]
        M = np.reshape(M,-1)
    return M

def prep_all_lats(M,indlev):
    if M.ndim == 4: #Ntime x Nlev x Nlat x Nlon
        M = M[:,indlev,:,:]
        M = M.swapaxes(1,3)
        M = np.reshape(M, (-1, M.shape[3])) #Ntime x Nlat x Nlon
    elif M.ndim ==3:
        M = np.reshape(M, (-1))[:,None]
    return M

def pack(d1,d2,axis=1):
    """Combines T & q profiles as an input matrix to NN"""
    return np.concatenate((d1,d2), axis=axis)

def unpack(data,vari,axis=1):
    """Reverse pack operation to turn ouput matrix into T & q"""
    N = int(data.shape[axis]/2)
    varipos = {'T':np.arange(N),'q':np.arange(N,2*N)}
    out = np.take(data,varipos[vari],axis=axis)
    return out

def pp(x, y, cv, num_samples, scaler_x=None, scaler_y=None):
    """Preprocess data by scaling and splitting it into 3 equally sized samples"""
    if num_samples==0: num_samples=x.shape[0]
    # Randomly choose samples
    samples = np.random.choice(x.shape[0], num_samples, replace=False)
    # Scale input data
    if scaler_x is None:
        # If no scaler given, create one and fit and transform the data
        scaler_x = preprocessing.MinMaxScaler(feature_range=(-1.0,1.0))
        x1,x2,x3    = _pp(x,  samples, scaler_x)
    else: 
        # If a scaler is given, use it to only transform the data
        x1,x2,x3    = _pp(x,  samples, scaler_x, fit_data=False)
    # Scale output data
    if scaler_y is None:
        # Since outputs are sparse, don't shift the mean
        scaler_y = preprocessing.MaxAbsScaler()
        y1,y2,y3    = _pp(y,  samples, scaler_y)
    else:
        y1,y2,y3    = _pp(y,  samples, scaler_y, fit_data=False)
    # We don't need to scale the classification of convection
    cv1,cv2,cv3 = _pp(cv, samples, None, transform_data=False)
    return scaler_x, scaler_y, x1, x2, x3, y1, y2, y3, cv1, cv2, cv3

def _pp(z, samples, scaler, transform_data=True, fit_data=True):
    """Preprocess data by scaling and splitting it into 3 equally sized samples"""
    # Scale data
    if (transform_data and fit_data):
        z = scaler.fit_transform(z)
    if (transform_data and not fit_data):
        z = scaler.transform(z)
    # Split data
    ss = np.floor(len(samples)/3) # number of samples in each set
    z1 = np.take(z,samples[   0:  ss], axis=0)
    z2 = np.take(z,samples[  ss:2*ss], axis=0)
    z3 = np.take(z,samples[2*ss:3*ss], axis=0)
    return z1,z2,z3

def limitrain(x,y,Pout,rainonly=False, verbose=True):
    indrain = np.greater(Pout, 0)
    if verbose:
        print('There is some amount of rain %.1f%% of the time' 
          %(100.*np.sum(indrain)/len(indrain)))
        print('There is a rate of >3 mm/day %.1f%% of the time' 
          %(100.*np.sum(np.greater(Pout*3600.*24.,3))/len(indrain)))
    if rainonly:
        x = x[indrain,:]
        y = y[indrain,:]
        Pout = Pout[indrain]
        if verbose:
            print('Only looking at times it is raining!')
    return x, y, Pout

def whenconvection(y, verbose=True):
    """Caluclate how often convection occurs...useful for classification
       Also store a variable that is 1 if convection and 0 if no convection"""
    cv = np.sum(np.abs(unpack(y, 'T')), axis=1)
    cv[cv > 0] = 1
    if verbose:
        print('There is convection %.1f%% of the time' %(100.*np.sum(cv)/len(cv)))
    return cv

def write_netcdf_twolayer(mlp,method,filename):
    # Grab weights and input normalization
    w1 = mlp.get_parameters()[0].weights
    w2 = mlp.get_parameters()[1].weights
    w3 = mlp.get_parameters()[2].weights
    b1 = mlp.get_parameters()[0].biases
    b2 = mlp.get_parameters()[1].biases
    b3 = mlp.get_parameters()[2].biases

    xscale_min = scaler_x.data_min_
    xscale_max = scaler_x.data_max_
    yscale_absmax = scaler_y.max_abs_

    # Write weights to file
    ncfile = Dataset(filename,'w')
    # Write the dimensions
    ncfile.createDimension('N_in',     w1.shape[0])
    ncfile.createDimension('N_h1',     w1.shape[1])
    ncfile.createDimension('N_h2',     w2.shape[1])
    ncfile.createDimension('N_out',    w3.shape[1])

    # Create variable entries in the file
    nc_w1 = ncfile.createVariable('w1',np.dtype('float64').char,('N_h1','N_in'    )) #Reverse dims
    nc_w2 = ncfile.createVariable('w2',np.dtype('float64').char,('N_h2','N_h1'     ))
    nc_w3 = ncfile.createVariable('w3',np.dtype('float64').char,('N_out','N_h2'    ))
    nc_b1 = ncfile.createVariable('b1',np.dtype('float64').char,('N_h1'))
    nc_b2 = ncfile.createVariable('b2',np.dtype('float64').char,('N_h2'))
    nc_b3 = ncfile.createVariable('b3',np.dtype('float64').char,('N_out'))
    if method == 'regress':
        nc_xscale_min = ncfile.createVariable('xscale_min',np.dtype('float64').char,('N_in'))
        nc_xscale_max = ncfile.createVariable('xscale_max',np.dtype('float64').char,('N_in'))
        nc_yscale_absmax = ncfile.createVariable('yscale_absmax',np.dtype('float64').char,('N_out'))
    # Write variables and close file - transpose because fortran reads it in "backwards"
    nc_w1[:] = w1.T
    nc_w2[:] = w2.T
    nc_w3[:] = w3.T
    nc_b1[:] = b1
    nc_b2[:] = b2
    nc_b3[:] = b3
    if method == 'regress':
        nc_xscale_min[:] = xscale_min
        nc_xscale_max[:] = xscale_max
        nc_yscale_absmax[:] = yscale_absmax
    ncfile.close()

def avg_hem(data, lat, axis, split=False):
    """Averages the NH and SH data (or splits them into two data sets)"""
    ixsh = np.where(lat<0)[0] # where returns a tuple
    ixnh = np.where(lat>=0)[0]
    if len(ixsh)==0:
        print(lat)
        ValueError('Appears that lat does not have SH values')
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

def load_one_lat(scaler_x, scaler_y, r_mlp, indlat, data_dir='./data/', minlev=0., rainonly=False):
    """Returns N_samples x 2*N_lev array of true and predicted values at a given latitude"""
    # Load data
    x, y, cv, Pout, lat, lev, dlev, timestep = loaddata(data_dir + 'nntest.nc', minlev,
                                                           rainonly=rainonly ,all_lats=False,
                                                           indlat=indlat, verbose=False)
    _, _, x1, x2, x3, y1, y2, y3, cv1, cv2, cv3 = pp(x, y, cv, 0, scaler_x=scaler_x, scaler_y=scaler_y)
    y3_pred=r_mlp.predict(x3)
    # Inverse transform back to physical units
    y3      = scaler_y.inverse_transform(y3)
    y3_pred = scaler_y.inverse_transform(y3_pred)
    T = unpack(y3,'T')
    q = unpack(y3,'q')
    T_pred = unpack(y3_pred,'T')
    q_pred = unpack(y3_pred,'q')
    return T, q, T_pred, q_pred

def stats_by_latlev(scaler_x, scaler_y, r_mlp, lat, lev):
    # Initialize
    Tmean = np.zeros((len(lat),len(lev)))
    qmean = np.zeros((len(lat),len(lev)))
    rmseT = np.zeros((len(lat),len(lev)))
    rmseq = np.zeros((len(lat),len(lev)))
    rT    = np.zeros((len(lat),len(lev)))
    rq    = np.zeros((len(lat),len(lev)))
    rmseT_lat = np.zeros(len(lat))
    rmseq_lat = np.zeros(len(lat))
    for i in range(len(lat)):
        T_true, q_true, T_pred, q_pred = load_one_lat(scaler_x, scaler_y, r_mlp, i)
        # Get means of true output
        Tmean[i,:] = np.mean(T_true,axis=0)
        qmean[i,:] = np.mean(q_true,axis=0)
        # Get rmse
        rmseT[i,:] = np.sqrt(metrics.mean_squared_error(T_true, T_pred, multioutput='raw_values'))
        rmseq[i,:] = np.sqrt(metrics.mean_squared_error(q_true, q_pred, multioutput='raw_values'))
        # Get correlation coefficients
        for j in range(len(lev)):
            rT[i,j], _ = scipy.stats.pearsonr(T_true[:,j], T_pred[:,j])
            rq[i,j], _ = scipy.stats.pearsonr(q_true[:,j], q_pred[:,j])

    return Tmean.T, qmean.T, rmseT.T, rmseq.T, rT.T, rq.T
