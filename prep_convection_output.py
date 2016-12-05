# Intended to be run on yellowstone: set the environment with the below
# commands:
# module load python
# module load all-python-libs

import numpy as np
from netCDF4 import Dataset
import pickle

# Data here is recorded once per day (not averaged, but a snapshot) and stored
# over a 1000 day period. N_lon_samp represents different longitudes we look at
# for a give longitude and time snapshot


time_beg = 1025
time_end = 4000
time_stp = 25
file_days = np.arange(time_beg, time_end, time_stp)
N_files = np.size(file_days)

# Hardcoded N_lev (N_lev) and N_lat (64) here
N_lev = 30
N_lat = 64
N_lon = 128
N_lon_samp = 5  # These are the number of samples (lon x time pairs) to take

Tin = np.zeros((time_stp, N_lev, N_lat, N_lon_samp, N_files))
qin = np.zeros((time_stp, N_lev, N_lat, N_lon_samp, N_files))
Tout = np.zeros((time_stp, N_lev, N_lat, N_lon_samp, N_files))
qout = np.zeros((time_stp, N_lev, N_lat, N_lon_samp, N_files))
Pout = np.zeros((time_stp, N_lat, N_lon_samp, N_files))
Tout_all = np.zeros((time_stp, N_lev, N_lat, N_lon_samp, N_files))
qout_all = np.zeros((time_stp, N_lev, N_lat, N_lon_samp, N_files))
Pout_all = np.zeros((time_stp, N_lat, N_lon_samp, N_files))
# Loop over files (stats stored daily)
for i, file_day in enumerate(file_days):
    # Initialzie
    zTin = np.zeros((time_stp, N_lev, N_lat, N_lon))
    zqin = np.zeros((time_stp, N_lev, N_lat, N_lon))
    zTout = np.zeros((time_stp, N_lev, N_lat, N_lon))
    zqout = np.zeros((time_stp, N_lev, N_lat, N_lon))
    zPout = np.zeros((time_stp, N_lat, N_lon))
    zTout_all = np.zeros((time_stp, N_lev, N_lat, N_lon))
    zqout_all = np.zeros((time_stp, N_lev, N_lat, N_lon))
    zPout_all = np.zeros((time_stp, N_lat, N_lon))
    # Set filename
    filename = '/glade/u/home/jdwyer/scratch/fms_output/' + \
        'del1.2_abs1.0_T42/history/day' + \
        str(file_day).zfill(4) + 'h00/day' + \
        str(file_day).zfill(4) + 'h00.1xday.nc'
    print(filename)
    # Open file and grab variables from it
    f = Dataset(filename, mode='r')
    # N_time x N_lev x N_lat x N_lon
    zTin = f.variables['tg_before_convection'][:]
    zqin = f.variables['qg_before_convection'][:]
    zTout = f.variables['dt_tg_convection'][:]
    zqout = f.variables['dt_qg_convection'][:]
    zPout = f.variables['convection_rain'][:]  # N_time x N_lat x N_lon
    zTout_all = zTout + f.variables['dt_tg_condensation'][:]
    zqout_all = zqout + f.variables['dt_qg_condensation'][:]
    zPout_all = zPout + f.variables['condensation_rain'][:]
    lat = f.variables['lat'][:]
    f.close()
    # Note that because of the way the data is now stored, we no longer need to
    # do any sort of shifting. Temperatures and humidities are explicitly
    # stored before convection is called.
    # N_time x N_lev x N_lat x N_lon -> N_time x N_lev x N_lat x N_lon x N_file
    # Loop over time steps in a given file
    for k in range(zTin.shape[0]):
        # Loop over latitudes
        for j in range(zTin.shape[2]):
            # Randomly choose a few longitudes
            ind_lon = np.random.randint(0, zTin.shape[3], N_lon_samp)
            # Numpy has some strange behavior when indexing and slicing are
            # combined. See: http://stackoverflow.com/q/27094438
            Tin[k, :, j, :, i] = zTin[k, :, j, :][:, ind_lon]
            qin[k, :, j, :, i] = zqin[k, :, j, :][:, ind_lon]
            Tout[k, :, j, :, i] = zTout[k, :, j, :][:, ind_lon]
            qout[k, :, j, :, i] = zqout[k, :, j, :][:, ind_lon]
            Tout_all[k, :, j, :, i] = zTout_all[k, :, j, :][:, ind_lon]
            qout_all[k, :, j, :, i] = zqout_all[k, :, j, :][:, ind_lon]
            Pout[k, j, :, i] = zPout[k, j, ind_lon]
            Pout_all[k, j, :, i] = zPout_all[k, j, ind_lon]

# Permute arrays to be N_lev x N_lat x N_lon_samp x N_file x N_time
permute = [1, 2, 3, 4, 0]
Tin = np.transpose(Tin, permute)
qin = np.transpose(qin, permute)
Tout = np.transpose(Tout, permute)
qout = np.transpose(qout, permute)
Tout_all = np.transpose(Tout_all, permute)
qout_all = np.transpose(qout_all, permute)
permute = [1, 2, 3, 0]
Pout = np.transpose(Pout, permute)
Pout_all = np.transpose(Pout_all, permute)

# Reshape array to be N_lev x N_lat x time_stp*N_samp*N_file
Tin = np.reshape(Tin, (N_lev, N_lat, -1))
qin = np.reshape(qin, (N_lev, N_lat, -1))
Tout = np.reshape(Tout, (N_lev, N_lat, -1))
qout = np.reshape(qout, (N_lev, N_lat, -1))
Pout = np.reshape(Pout, (N_lat, -1))
Tout_all = np.reshape(Tout_all, (N_lev, N_lat, -1))
qout_all = np.reshape(qout_all, (N_lev, N_lat, -1))
Pout_all = np.reshape(Pout_all, (N_lat, -1))

# Convert heating rates from K/s to K/day and from kg/kg/s to g/kg/day
Tout = Tout * 3600 * 24
qout = qout * 3600 * 24 * 1000
Tout_all = Tout_all * 3600 * 24
qout_all = qout_all * 3600 * 24 * 1000
# Convert precip from kg/m/m/s to mm/day
Pout = Pout * 3600 * 24
Pout_all = Pout_all * 3600 * 24

# Shuffle data and store it in separate training and validation files
N_trn_exs = Tin.shape[2]
randinds = np.random.permutation(N_trn_exs)
i70 = int(0.7*np.size(randinds))
i90 = int(0.9*np.size(randinds))
randind_trn = randinds[:i70]
randind_tst = randinds[i70:i90]
randind_vld = randinds[i90:]
# Store the data in files
# For convection-only learning
pickle.dump([Tin[:, :, randind_trn], qin[:, :, randind_trn],
             Tout[:, :, randind_trn], qout[:, :, randind_trn],
             Pout[:, randind_trn], lat], open('./conv_training_v3.pkl', 'wb'))
pickle.dump([Tin[:, :, randind_tst], qin[:, :, randind_tst],
             Tout[:, :, randind_tst], qout[:, :, randind_tst],
             Pout[:, randind_tst], lat], open('./conv_testing_v3.pkl', 'wb'))
pickle.dump([Tin[:, :, randind_vld], qin[:, :, randind_vld],
             Tout[:, :, randind_vld], qout[:, :, randind_vld],
             Pout[:, randind_vld], lat],
            open('./conv_validation_v3.pkl', 'wb'))
# For convection + condensation learning
pickle.dump([Tin[:, :, randind_trn], qin[:, :, randind_trn],
             Tout_all[:, :, randind_trn], qout_all[:, :, randind_trn],
             Pout_all[:, randind_trn], lat],
            open('./convcond_training_v3.pkl', 'wb'))
pickle.dump([Tin[:, :, randind_tst], qin[:, :, randind_tst],
             Tout_all[:, :, randind_tst], qout_all[:, :, randind_tst],
             Pout_all[:, randind_tst], lat],
            open('./convcond_testing_v3.pkl', 'wb'))
pickle.dump([Tin[:, :, randind_vld], qin[:, :, randind_vld],
             Tout_all[:, :, randind_vld], qout_all[:, :, randind_vld],
             Pout_all[:, randind_vld], lat],
            open('./convcond_validation_v3.pkl', 'wb'))
