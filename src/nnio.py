import src.nnload as nnload
from netCDF4 import Dataset
import numpy as np
import pickle


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
        w1[:,:,i] = mlp.get_parameters()[0].weights
        w2[:,:,i] = mlp.get_parameters()[1].weights
        b1[:,i] = mlp.get_parameters()[0].biases
        b2[:,i] = mlp.get_parameters()[1].biases
        xscale_mean[:,i] = x_pp.mean_
        xscale_stnd[:,i] = x_pp.scale_
        Nlev = len(lev)
        yscale_absmax[:Nlev,i] = y_pp[0]
        yscale_absmax[Nlev:,i] = y_pp[1]
    # Write weights to file
    ncfile = Dataset(filename, 'w')
    # Write the dimensions
    ncfile.createDimension('N_in', w1.shape[0])
    ncfile.createDimension('N_h1', w1.shape[1])
    ncfile.createDimension('N_out', w2.shape[1])
    ncfile.createDimension('N_e', N_e)
    # Create variable entries in the file
    nc_w1 = ncfile.createVariable('w1', np.dtype('float64').char,
                                  ( 'N_e','N_h1','N_in'))  # Reverse dims
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
    # Write global file attributes
    # ncfile.description = mlp_str
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
