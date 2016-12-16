import src.nnload as nnload
from netCDF4 import Dataset
import numpy as np
import pickle


def build_training_dataset(expt, t_step, t_beg, t_end, N_lon_samp=5):
    """Builds training, testing, and cross-validation datasets from an
       idealized GCM run folder. Assumes tendencies are stored for
       instantaneous values once per day. Also assumes T42 resolution
       If running on yellowstone be sure to load "python" and "all-python-libs"
       modules.
       Note that because of the way the data is now stored, we no longer need
       to do any sort of shifting. Temperatures and humidities are explicitly
       stored before convection is called.
    Args:
     expt (str): Path to the experiment folder
     t_step (int): Number of days between when each file is saved
     t_beg (int): Date of first time being saved
     t_end (int): Date of last time being saved
     N_lon_samp (int): Number of random longitude samples to take at each lat
                       at each time step. Default value is 5 (T42 resolution)
    """
    file_days = np.arange(t_beg, t_end, t_step)  # file_days = [1010]
    N_files = np.size(file_days)
    # Hardcoded values here
    N_lev = 30
    N_lat = 64
    N_lon = 128
    # Initialize
    Tin = np.zeros((t_step, N_lev, N_lat, N_lon_samp, N_files))
    qin = np.zeros((t_step, N_lev, N_lat, N_lon_samp, N_files))
    Tout = np.zeros((t_step, N_lev, N_lat, N_lon_samp, N_files))
    qout = np.zeros((t_step, N_lev, N_lat, N_lon_samp, N_files))
    Pout = np.zeros((t_step, N_lat, N_lon_samp, N_files))
    Tout_all = np.zeros((t_step, N_lev, N_lat, N_lon_samp, N_files))
    qout_all = np.zeros((t_step, N_lev, N_lat, N_lon_samp, N_files))
    Pout_all = np.zeros((t_step, N_lat, N_lon_samp, N_files))
    # Loop over files in experiment folder (assumes stats stored daily)
    for i, file_day in enumerate(file_days):
        # Initialzie
        zTin = np.zeros((t_step, N_lev, N_lat, N_lon))
        zqin = np.zeros((t_step, N_lev, N_lat, N_lon))
        zTout = np.zeros((t_step, N_lev, N_lat, N_lon))
        zqout = np.zeros((t_step, N_lev, N_lat, N_lon))
        zPout = np.zeros((t_step, N_lat, N_lon))
        zTout_all = np.zeros((t_step, N_lev, N_lat, N_lon))
        zqout_all = np.zeros((t_step, N_lev, N_lat, N_lon))
        zPout_all = np.zeros((t_step, N_lat, N_lon))
        # Get filename
        filename = '/glade/u/home/jdwyer/scratch/fms_output/' + \
            expt + '/history/day' + \
            str(file_day).zfill(4) + 'h00/day' + \
            str(file_day).zfill(4) + 'h00.1xday.nc'
        print(filename)
        # Open file and grab variables from it
        f = Dataset(filename, mode='r')
        # N_time x N_lev x N_lat x N_lon
        zTin = f.variables['t_intermed'][:]
        zqin = f.variables['q_intermed'][:]
        zTout = f.variables['dt_tg_convection'][:]
        zqout = f.variables['dt_qg_convection'][:]
        zPout = f.variables['convection_rain'][:]  # N_time x N_lat x N_lon
        zTout_all = zTout + f.variables['dt_tg_condensation'][:]
        zqout_all = zqout + f.variables['dt_qg_condensation'][:]
        zPout_all = zPout + f.variables['condensation_rain'][:]
        lat = f.variables['lat'][:]
        f.close()
        # N_time x N_lev x N_lat x N_lon ->
        # N_time x N_lev x N_lat x N_lon x N_file
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
    # Reshape array to be N_lev x N_lat x t_step*N_samp*N_file
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
                 Pout[:, randind_trn], lat],
                open('./' + expt + '_conv_training.pkl', 'wb'))
    pickle.dump([Tin[:, :, randind_tst], qin[:, :, randind_tst],
                 Tout[:, :, randind_tst], qout[:, :, randind_tst],
                 Pout[:, randind_tst], lat],
                open('./' + expt + '_conv_testing.pkl', 'wb'))
    pickle.dump([Tin[:, :, randind_vld], qin[:, :, randind_vld],
                 Tout[:, :, randind_vld], qout[:, :, randind_vld],
                 Pout[:, randind_vld], lat],
                open('./' + expt + '_conv_validation.pkl', 'wb'))
    # For convection + condensation learning
    pickle.dump([Tin[:, :, randind_trn], qin[:, :, randind_trn],
                 Tout_all[:, :, randind_trn], qout_all[:, :, randind_trn],
                 Pout_all[:, randind_trn], lat],
                open('./' + expt + '_convcond_training.pkl', 'wb'))
    pickle.dump([Tin[:, :, randind_tst], qin[:, :, randind_tst],
                 Tout_all[:, :, randind_tst], qout_all[:, :, randind_tst],
                 Pout_all[:, randind_tst], lat],
                open('./' + expt + '_convcond_testing.pkl', 'wb'))
    pickle.dump([Tin[:, :, randind_vld], qin[:, :, randind_vld],
                 Tout_all[:, :, randind_vld], qout_all[:, :, randind_vld],
                 Pout_all[:, randind_vld], lat],
                open('./' + expt + '_convcond_validation.pkl', 'wb'))


def write_netcdf_v4():
    mlp_str = 'X-StandardScaler-qTindi_Y-SimpleY-qTindi_' + \
        'Ntrnex100000_r_100R_mom0.9reg1e-06_Niter10000_v3'
    datasource = './data/conv_training_v3.pkl'
    # Set output filename
    filename = '/Users/jgdwyer/neural_weights_v4.nc'
    # Load ANN and preprocessors
    mlp, _, errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, dlev = \
        pickle.load(open('./data/regressors/' + mlp_str + '.pkl', 'rb'))
    # Need to transform some data for preprocessors to be able to export params
    x_unscl, y_unscl, _, _, _, _, _, _ = nnload.loaddata(datasource,
                                                         minlev=min(lev))
    x_scl = nnload.transform_data(x_ppi, x_pp, x_unscl)
    _ = nnload.transform_data(y_ppi, y_pp, y_unscl)
    # Also need to use the predict method to be able to export ANN params
    _ = mlp.predict(x_scl)
    # Grab weights and input normalization
    w1 = mlp.get_parameters()[0].weights
    w2 = mlp.get_parameters()[1].weights
    b1 = mlp.get_parameters()[0].biases
    b2 = mlp.get_parameters()[1].biases
    xscale_mean = x_pp.mean_
    xscale_stnd = x_pp.scale_
    Nlev = len(lev)
    yscale_absmax = np.zeros(b2.shape)
    yscale_absmax[:Nlev] = y_pp[0]
    yscale_absmax[Nlev:] = y_pp[1]
    # Write weights to file
    ncfile = Dataset(filename, 'w')
    # Write the dimensions
    ncfile.createDimension('N_in', w1.shape[0])
    ncfile.createDimension('N_h1', w1.shape[1])
    ncfile.createDimension('N_out', w2.shape[1])
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


def write_netcdf_ensemble1():
    ntrns = np.arange(125000, 125010)
    base1 = 'X-StandardScaler-qTindi_Y-SimpleY-qTindi_Ntrnex'
    base2 = '_r_50R_mom0.9reg1e-06_Niter3000_v3'
    mlp_str = [base1 + str(ntrn) + base2 for ntrn in ntrns]
    N_e = len(mlp_str)
    datasource = './data/conv_training_v3.pkl'
    # Set output filename
    filename = '/Users/jgdwyer/neural_weights_ensemble1.nc'
    # Load ANN and preprocessors
    yscale_absmax = np.zeros((32, len(mlp_str)))
    w1 = np.zeros((32, 50, N_e))
    w2 = np.zeros((50, 32, N_e))
    b1 = np.zeros((50, N_e))
    b2 = np.zeros((32, N_e))
    xscale_mean = np.zeros((32, N_e))
    xscale_stnd = np.zeros((32, N_e))
    yscale_absmax = np.zeros((32, N_e))

    for i in range(len(mlp_str)):
        mlp, _, errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, dlev = \
            pickle.load(open('./data/regressors/' + mlp_str[i] + '.pkl', 'rb'))
        # Need to transform some data for preprocessors to be able to export
        # params
        x_unscl, y_unscl, _, _, _, _, _, _ = nnload.loaddata(datasource,
                                                             minlev=min(lev))
        x_scl = nnload.transform_data(x_ppi, x_pp, x_unscl)
        y_scl = nnload.transform_data(y_ppi, y_pp, y_unscl)
        # Also need to use the predict method to be able to export ANN params
        _ = mlp.predict(x_scl)
        # Grab weights and input normalization
        w1[:, :, i] = mlp.get_parameters()[0].weights
        w2[:, :, i] = mlp.get_parameters()[1].weights
        b1[:, i] = mlp.get_parameters()[0].biases
        b2[:, i] = mlp.get_parameters()[1].biases
        xscale_mean[:, i] = x_pp.mean_
        xscale_stnd[:, i] = x_pp.scale_
        Nlev = len(lev)
        yscale_absmax[:Nlev, i] = y_pp[0]
        yscale_absmax[Nlev:, i] = y_pp[1]
    # Write weights to file
    ncfile = Dataset(filename, 'w')
    # Write the dimensions
    ncfile.createDimension('N_in', w1.shape[0])
    ncfile.createDimension('N_h1', w1.shape[1])
    ncfile.createDimension('N_out', w2.shape[1])
    ncfile.createDimension('N_e', N_e)
    # Create variable entries in the file
    # Variables need to "reversed" to be read in by Fortran GCM code
    nc_w1 = ncfile.createVariable('w1', np.dtype('float64').char,
                                  ('N_e', 'N_h1', 'N_in'))
    nc_w2 = ncfile.createVariable('w2', np.dtype('float64').char,
                                  ('N_e', 'N_out', 'N_h1'))
    nc_b1 = ncfile.createVariable('b1', np.dtype('float64').char,
                                  ('N_e', 'N_h1'))
    nc_b2 = ncfile.createVariable('b2', np.dtype('float64').char,
                                  ('N_e', 'N_out'))
    nc_xscale_mean = ncfile.createVariable('xscale_mean',
                                           np.dtype('float64').char,
                                           ('N_e', 'N_in'))
    nc_xscale_stnd = ncfile.createVariable('xscale_stnd',
                                           np.dtype('float64').char,
                                           ('N_e', 'N_in'))
    nc_yscale_absmax = ncfile.createVariable('yscale_absmax',
                                             np.dtype('float64').char,
                                             ('N_e', 'N_out'))
    # Write variables and close file - transpose because fortran reads it in
    # "backwards"
    nc_w1[:] = np.transpose(w1, (2, 1, 0))
    nc_w2[:] = np.transpose(w2, (2, 1, 0))
    nc_b1[:] = b1.T
    nc_b2[:] = b2.T
    nc_xscale_mean[:] = xscale_mean.T
    nc_xscale_stnd[:] = xscale_stnd.T
    nc_yscale_absmax[:] = yscale_absmax.T
    ncfile.close()


def write_netcdf_convcond_v1():
    mlp_str = 'convcond_X-StandardScaler-qTindi_Y-SimpleY-qTindi_' +\
        'Ntrnex100000_r_100R_mom0.9reg1e-05_Niter10000_v3'
    datasource = './data/convcond_training_v3.pkl'
    # Set output filename
    filename = '/Users/jgdwyer/neural_weights_convcond_v1.nc'
    # Load ANN and preprocessors
    mlp, _, errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, dlev = \
        pickle.load(open('./data/regressors/' + mlp_str + '.pkl', 'rb'))
    # Need to transform some data for preprocessors to be able to export params
    x_unscl, y_unscl, _, _, _, _, _, _ = nnload.loaddata(datasource,
                                                         minlev=min(lev))
    x_scl = nnload.transform_data(x_ppi, x_pp, x_unscl)
    y_scl = nnload.transform_data(y_ppi, y_pp, y_unscl)
    # Also need to use the predict method to be able to export ANN params
    _ = mlp.predict(x_scl)
    # Grab weights and input normalization
    w1 = mlp.get_parameters()[0].weights
    w2 = mlp.get_parameters()[1].weights
    b1 = mlp.get_parameters()[0].biases
    b2 = mlp.get_parameters()[1].biases
    xscale_mean = x_pp.mean_
    xscale_stnd = x_pp.scale_
    Nlev = len(lev)
    yscale_absmax = np.zeros(b2.shape)
    yscale_absmax[:Nlev] = y_pp[0]
    yscale_absmax[Nlev:] = y_pp[1]
    # Write weights to file
    ncfile = Dataset(filename, 'w')
    # Write the dimensions
    ncfile.createDimension('N_in', w1.shape[0])
    ncfile.createDimension('N_h1', w1.shape[1])
    ncfile.createDimension('N_out', w2.shape[1])
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


def verify_netcdf_weights():
    r_str = 'convcond_X-StandardScaler-qTindi_Y-SimpleY-qTindi_' +\
        'Ntrnex100000_r_100R_mom0.9reg1e-05_Niter10000_v3'
    nc_str = '/Users/jgdwyer/neural_weights_convcond_v1.nc'
    # Load unscaled data
    x, y, cv, Pout, lat, lev, dlev, timestep = \
        nnload.loaddata('./data/convcond_testing_v3.pkl',
                        0.2, all_lats=True, indlat=None, rainonly=False)
    # Load preprocessers
    r_mlp_eval, _, errors, x_ppi, y_ppi, x_pp, y_pp, lat2, lev2, dlev = \
        pickle.load(open('./data/regressors/' + r_str + '.pkl', 'rb'))
    print('Loading predictor: ' + r_str)
    print('Loading ncfile :' + nc_str)
    # Load netcdf files
    ncfile = Dataset(nc_str, 'r')
    yscale_absmax = ncfile['yscale_absmax'][:]
    yscale_absmax = yscale_absmax[:, None].T
    xscale_mean = ncfile['xscale_mean'][:]
    xscale_mean = xscale_mean[:, None].T
    xscale_std = ncfile['xscale_stnd'][:]
    xscale_std = xscale_std[:, None].T
    print(x_ppi)
    # Scaled variables as calculated by NN weights
    xs = nnload.transform_data(x_ppi, x_pp, x)
    ys = nnload.transform_data(y_ppi, y_pp, y)
    # Scaled variables as calculated by hand from netcdf files
    xs_byhand = (x - xscale_mean)/xscale_std
    ys_byhand = y/yscale_absmax
    print('Difference between x-scaling methods: {:.1f}'.
          format(np.sum(np.abs(xs - xs_byhand))))
    print('Difference between y-scaling methods: {:.1f}'.
          format(np.sum(np.abs(ys - ys_byhand))))
    # Now check that transformation is done correctly
    # Load NN weights
    w1 = ncfile['w1'][:].T
    w2 = ncfile['w2'][:].T
    b1 = ncfile['b1'][:]
    b2 = ncfile['b2'][:]
    yps_byhand = np.dot(xs_byhand, w1) + b1
    yps_byhand[yps_byhand < 0] = 0
    yps_byhand = np.dot(yps_byhand, w2) + b2
    yps = r_mlp_eval.predict(xs)
    print('Difference between predicted tendencies: {:.1f}'.
          format(np.sum(np.abs(yps - yps_byhand))))


def compare_convcond_prediction(cv_str, cvcd_str, minlev):
    cv_mlp, _, errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, _ = \
        pickle.load(open('./data/regressors/' + cv_str + '.pkl', 'rb'))
    cvcd_mlp, _, errors, x_ppi_check, y_ppi_check, x_pp, y_pp, lat, lev, _ = \
        pickle.load(open('./data/regressors/' + cvcd_str + '.pkl', 'rb'))
    # Check that preprocessers are the same
    if ((x_ppi != x_ppi_check) or (y_ppi != y_ppi_check)):
        raise ValueError('Preprocessing schemes different for conv only and ' +
                         'conv+cond! This means that comparing the two in ' +
                         'scaled space may give different results')
    # Load data
    x_unscl, ytcv_unscl, _, _, _, _, _, _ = \
        nnload.loaddata('./data/conv_testing_v3.pkl', minlev=minlev,
                        N_trn_exs=10000, randseed=True)
    xcvcd_unscl, ytcvcd_unscl, _, _, _, _, _, _ = \
        nnload.loaddata('./data/convcond_testing_v3.pkl', minlev=minlev,
                        N_trn_exs=10000, randseed=True)
    # Check that x values are the same to make sure random seeds are same
    if np.sum(np.abs(x_unscl - xcvcd_unscl)) > 0.0:
        raise ValueError('Data loaded in different order!')
    # Convert true y-values to scaled by applying an inverse transformation
    ytcv_scl = nnload.transform_data(y_ppi, y_pp, ytcv_unscl)
    ytcvcd_scl = nnload.transform_data(y_ppi, y_pp, ytcvcd_unscl)
    # Derived true y-values for cond only
    ytcd_scl = ytcvcd_scl - ytcvcd_scl
    # Calculate predicted y values for conv and convcond
    ypcv_scl = cv_mlp.predict(x_scl)
    ypcvcd_scl = cvcd_mlp.predict(x_scl)
    # Add true cond values to ycv_true and ycv_pred
    v = 'q'
    mse_cvcd_predictboth = nnload.calc_mse(nnload.unpack(ypcvcd_scl, v),
                                           nnload.unpack(ytcvcd_scl, v),
                                           relflag=True)
    mse_cv = nnload.calc_mse(nnload.unpack(ypcv_scl, v),
                             nnload.unpack(ytcv_scl, v), relflag=True)
    print('MSE predicting convection and condensation in one step: {:.5f}'.
          format(mse_cvcd_predictboth))
    print('MSE predicting convection only (no condensation): {:.5f}'.
          format(mse_cv))
