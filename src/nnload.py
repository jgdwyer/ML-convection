import numpy as np
from netCDF4 import Dataset

def loaddata(filename, minlev):
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
    # Apply preprocessing to data
    Tin =  prep_all_lats(Tin ,indlev)
    qin =  prep_all_lats(qin ,indlev)
    Tout = prep_all_lats(Tout,indlev)
    qout = prep_all_lats(qout,indlev)
    Pout = prep_all_lats(Pout,indlev)
    # Convert heating rates to K/day and g/kg/day
    Tout = Tout*3600.*24.
    qout = qout*3600.*24.*1000.
    # Concatenate input and output variables together
    x = pack(Tin,  qin , axis=1)
    y = pack(Tout, qout, axis=1)
    return (x, y, Pout, lat, lev, dlev, timestep)

def prep(M,indlev,indlat):
    indlat = np.logical_and(lat>46,lat<47)
    if M.ndim == 4:
        M = M[:,indlev,indlat,:].squeeze()
    elif M.ndim == 3:
        M = M[:,indlat,:]
    M = M.swapaxes(1,2) # N_time x N_lon x N_lev 
    M = np.reshape(M, (-1, M.shape[2])) #Now N_time*N_lon x N_lev
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
    return np.concatenate((d1,d2), axis=axis)

def unpack(data,vari,axis=1):
    N = int(data.shape[axis]/2)
    varipos = {'T':np.arange(N),'q':np.arange(N,2*N)}
    out = np.take(data,varipos[vari],axis=axis)
    return out

def pp(z,samples,scaler,scale_data=True):
    """Preprocess data by scaling and splitting it into 3 equally sized samples"""
    # Scale data
    if scale_data:
        z = scaler.fit_transform(z)
    # Split data
    ss = 10000 # number of samples in each set
    z1 = np.take(z,samples[   0:  ss], axis=0)
    z2 = np.take(z,samples[  ss:2*ss], axis=0)
    z3 = np.take(z,samples[2*ss:3*ss], axis=0)
    return z1,z2,z3

def limitrain(x,y,Pout,rainonly=False):
    indrain = np.greater(Pout, 0)
    print('There is some amount of rain %.1f%% of the time' 
          %(100.*np.sum(indrain)/len(indrain)))
    print('There is a rate of >3 mm/day %.1f%% of the time' 
          %(100.*np.sum(np.greater(Pout*3600.*24.,3))/len(indrain)))
    if rainonly:
        x = x[indrain,:]
        y = y[indrain,:]
        Pout = Pout[indrain]
        print('Only looking at times it is raining!')
    return x, y, Pout

def whenconvection(y):
    """Caluclate how often convection occurs...useful for classification
       Also store a variable that is 1 if convection and 0 if no convection"""
    cv = np.sum(np.abs(unpack(y, 'T')), axis=1)
    cv[cv > 0] = 1
    print('There is convection %.1f%% of the time' %(100.*np.sum(cv)/len(cv)))
    return cv
