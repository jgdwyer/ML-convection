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
    
