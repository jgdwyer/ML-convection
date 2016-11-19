#module load python
#module load all-python-libs

import numpy as np
from netCDF4 import Dataset
import pickle

# These are the number of samples (lon x time pairs) to take 
# for each 24-hour period at each latitude
N_samples = 10
first_day = 1001
last_day  = 1051 #not inclusive
N_days = last_day - first_day


# Hardcoded N_lev and N_lat here
Tin  = np.zeros((30,64,N_days,N_samples))
qin  = np.zeros((30,64,N_days,N_samples))
Tout = np.zeros((30,64,N_days,N_samples))
qout = np.zeros((30,64,N_days,N_samples))
Pout = np.zeros((   64,N_days,N_samples))

# Loop over files (stats stored daily)
for i in range(first_day,last_day):
    ii = i-first_day
    filename='/glade/u/home/jdwyer/scratch/fms_output/' + \
             'del1.2_abs1.0_T42_30min/history/day' + \
              str(i).zfill(4) + 'h00/day' + \
              str(i).zfill(4) + 'h00.timestep.nc'
    print(filename)
    f = Dataset(filename,mode='r')
    zTin = f.variables['temp'][:] # N_time x N_lev x N_lat x N_lon
    zqin = f.variables['sphum'][:]
    zTout= f.variables['dt_tg_convection'][:]
    zqout= f.variables['dt_qg_convection'][:]
    zPout= f.variables['convection_rain'][:] # N_time x N_lat x N_lon
    lat = f.variables['lat'][:]
    f.close()
    # The part of the timestep in the model when these fields are saved is different
    # Shift them so that input happens BEFORE output
    zTin = zTin[1:,:,:,:]
    zqin = zqin[1:,:,:,:]
    zTout = zTout[0:-1,:,:,:]
    zqout = zqout[0:-1,:,:,:]
    zPout = zPout[0:-1,:,:]
    # Convert heating rates from K/s to K/day and from kg/kg/s to g/kg/day
    zTout = zTout*3600*24
    zqout = zqout*3600*24*1000
    #Now loop over each latitude and choose selections at random
    for j,_ in enumerate(lat):
        ind_time = np.random.randint(0,Tin.shape[0],10)
        ind_lon  = np.random.randint(0,Tin.shape[3],10)
        count = 0
        for k,_ in enumerate(ind_time):
            Tin[:,j,ii,count]  = zTin[ind_time[k],:,j,ind_lon[k]] # N_lev x N_lat x N_days x N_samp
            qin[:,j,ii,count]  = zqin[ind_time[k],:,j,ind_lon[k]]
            Tout[:,j,ii,count] = zTout[ind_time[k],:,j,ind_lon[k]]
            qout[:,j,ii,count] = zqout[ind_time[k],:,j,ind_lon[k]]
            Pout[j,ii,count]   = zPout[ind_time[k],  j,ind_lon[k]]
            count += 1
        
# Reshape array to be N_lev x N_lat x N_days*N_samp
Tin  = np.reshape(Tin ,(Tin.shape[0] ,Tin.shape[1] ,-1))
qin  = np.reshape(qin ,(qin.shape[0] ,qin.shape[1] ,-1))
Tout = np.reshape(Tout,(Tout.shape[0],Tout.shape[1],-1))
qout = np.reshape(qout,(qout.shape[0],qout.shape[1],-1))
Pout = np.reshape(Pout,(Pout.shape[1],              -1))
# Store the files
pickle.dump([Tin, qin, Tout, qout, Pout, lat],  open('./convection_50day.pkl', 'wb'))



