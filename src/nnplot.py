import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import src.nnload as nnload
import scipy.stats 
from sklearn import metrics, preprocessing
import pickle
import os
unpack = nnload.unpack

# ---   META PLOTTING SCRIPTS  --- #
#def plots_by_lat(pp_x, pp_y, r_mlp, lat):

def plot_all_figs(r_str, x_scl, ytrue_scl):
    r_mlp_eval, _, errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, dlev = pickle.load(open('./data/regressors/' + r_str + '.pkl', 'rb'))
    # Set figure path and create directory if it does not exist
    figpath = './figs/' + r_str + '/'
    if not os.path.exists(figpath):
        os.makedirs(figpath)

    # Plot model errors over iteration history
    plot_model_error_over_time(errors, r_str, figpath)

    # Inverse transform to get unscaled (untransformed) data in physical units
    x_unscl     = nnload.inverse_transform_data(x_ppi, x_pp, x_scl)
    ytrue_unscl = nnload.inverse_transform_data(y_ppi, y_pp, ytrue_scl)

    # Use algorithm to get predicted output
    ypred_scl   = r_mlp_eval.predict(x_scl)
    ypred_unscl = nnload.inverse_transform_data(y_ppi, y_pp, ypred_scl)

    # Plot historgram showing how scaling changed character of input and output data
    check_scaling_distribution(x_unscl, x_scl, ytrue_unscl, ytrue_scl, lat, lev, figpath)
 
    # Plot histogram showing how well true and predicted values match
    check_output_distribution(ytrue_unscl, ytrue_scl, ypred_unscl, ypred_scl, lat, lev, figpath)

    # Plot means and standard deviations
    plot_means_stds(ytrue_unscl, ypred_unscl, lev, figpath)

    # Plot correlation coefficient, explained variance, and rmse
    plot_error_stats(ytrue_unscl, ypred_unscl, lev, figpath)

    # Plot a time series of precipitaiton
    plot_precip(ytrue_unscl, ypred_unscl, dlev, figpath)

    # Plot the enthalpy conservation
    plot_enthalpy(ytrue_unscl, ypred_unscl, dlev, figpath)

    # Plot mean, bias, rmse, r^  (lat vs lev)
    make_contour_plots(figpath, x_ppi, y_ppi, x_pp, y_pp, r_mlp_eval, lat, lev)

    # Plot the rmse vs lat
    #nnplot.plot_rmse_vs_lat(r

def make_contour_plots(figpath, x_ppi, y_ppi, x_pp, y_pp, r_mlp_eval, lat, lev):
    Tmean, qmean, Tbias, qbias, rmseT, rmseq, rT, rq = nnload.stats_by_latlev(x_ppi, y_ppi, x_pp, y_pp, r_mlp_eval, lat, lev)
    # Make figs
    # True means
    f,ax1,ax2 = plot_contour(Tmean,qmean,lat,lev, avg_hem=True)
    ax1.set_title('Temperature True Mean [K/day]')
    ax2.set_title('Humidity True Mean [kg/kg/day]')
    # Bias from true mean
    f,ax1,ax2 = plot_contour(Tbias,qbias,lat,lev, avg_hem=True)
    ax1.set_title('Temperature Mean Bias [K/day]')
    ax2.set_title('Humidity Mean Bias [kg/kg/day]')
    # Root mean squared error
    f,ax1,ax2 = plot_contour(rmseT,rmseq,lat,lev, avg_hem=True)
    ax1.set_title('Temperature RMSE [K/day]')
    ax2.set_title('Humidity RMSE [kg/kg/day]')
    # Pearson r Correlation Coefficient
    f,ax1,ax2 = plot_contour(rT, rq, lat, lev, avg_hem=True)
    ax1.set_title('Temperature CorrelationCoefficient')
    ax2.set_title('Humidity Correlation Coefficient')

def plot_contour(T, q, lat, lev, avg_hem=False):
    if avg_hem:
        T, _ = nnload.avg_hem(T, lat, 1)
        q, lat = nnload.avg_hem(q, lat, 1)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    cax1 = ax1.contourf(lat, lev, T)
    ax1.set_ylim(1,0)
    ax1.set_ylabel(r'$\sigma$')
    f.colorbar(cax1,ax=ax1)
    cax2 = ax2.contourf(lat, lev, q)
    ax2.set_ylim(1,0)
    ax2.set_ylabel(r'$\sigma$')
    f.colorbar(cax2,ax=ax2)
    ax2.set_xlabel('Latitude')
    return f, ax1, ax2 # return figure handle

# Plot rmse vs lat
# NEED TO FIX THIS SCRIPT
def plot_rmse_vs_lat(r_mlp, figpath, data_dir='./data/', minlev=0.0, rainonly=False):
    #rmseT_nh, rmseT_sh, rmseq_nh, rmseq_sh, lat = _plot_rmse_vs_lat(r_mlp,data_dir='./data/',minlev=0.0,rainonly=False)
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(lat, rmseT_nh,label='NH')
    plt.plot(lat, rmseT_sh,label='SH')
    plt.title('Temp column RMSE')
    plt.subplot(2,1,2)
    plt.plot(lat, rmseq_nh, label='NH')
    plt.plot(lat, rmseq_sh, label='SH')
    plt.title('Humidity column RMSE')
    plt.xlabel('Latitude')
    plt.legend()
    fig.savefig(figpath + 'rmse_vs_lat.png', bbox_inches='tight',dpi=450)
# Plot means and standard deviations
def plot_means_stds(y3_true, y3_pred, lev, figpath):
    fig = plt.figure()
    do_mean_or_std('mean','T',y3_true,y3_pred, lev, 1)
    do_mean_or_std('mean','q',y3_true,y3_pred, lev, 2)
    do_mean_or_std('std','T',y3_true,y3_pred, lev, 3)
    do_mean_or_std('std','q',y3_true,y3_pred, lev, 4)
    fig.savefig(figpath + 'regress_means_stds.png', bbox_inches='tight',dpi=450)

# Plot correlation coefficient, explained variance, and rmse
def plot_error_stats(y3_true, y3_pred, lev, figpath):
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plot_pearsonr(y3_true, y3_pred, 'T', lev, label='T')
    plot_pearsonr(y3_true, y3_pred, 'q', lev, label='q')
    plt.legend(loc="upper left")
    plt.subplot(2,2,2)
    plot_expl_var(y3_true, y3_pred, 'T', lev)
    plot_expl_var(y3_true, y3_pred, 'q', lev)
    plt.subplot(2,2,3)
    plot_rmse(y3_true, y3_pred, 'T', lev)
    plt.subplot(2,2,4)
    plot_rmse(y3_true, y3_pred, 'q', lev)
    fig.savefig(figpath + 'regress_stats.png',bbox_inches='tight',dpi=450)

#Plot a time series of precipitaiton
def plot_precip(y3_true, y3_pred, dlev, figpath):
    fig = plt.figure()
    _plot_precip(y3_true,y3_pred, dlev)
    fig.savefig(figpath + 'regress_P_rate.png',bbox_inches='tight',dpi=450)

# Plot the enthalpy conservation
def plot_enthalpy(y3_true, y3_pred, dlev, figpath):
    fig = plt.figure(5)
    plt.subplot(2,1,1)
    _plot_enthalpy(y3_true, dlev, label='true')
    plt.legend(loc="upper left")
    plt.subplot(2,1,2)
    _plot_enthalpy(y3_pred, dlev, label='predict')
    plt.legend(loc="upper left")
    fig.savefig(figpath + 'regress_enthalpy.png',bbox_inches='tight',dpi=450)


# ----  PLOTTING SCRIPTS  ---- # 
out_str_dict = {'T':'K/day','q':'g/kg/day'}

def do_mean_or_std(method,vari,in1,in3,lev,ind):
    methods = {'mean':np.mean,'std':np.std}
    plt.subplot(2,2,ind)
    m = lambda x: methods[method](unpack(x,vari), axis=0).T
    plt.plot(m(in1), lev, label='true')
    plt.plot(m(in3), lev, label='pred')
    plt.ylim(np.amax(lev),np.amin(lev))
    plt.ylabel('$\sigma$')
    plt.xlabel(out_str_dict[vari])
    plt.title(vari + " " + method)
    plt.legend()
        
def plot_pearsonr(y_true,y_pred,vari,lev,label=None):
    r = np.empty(y_true.shape[1])
    prob = np.empty(y_true.shape[1])
    for i in range(y_true.shape[1]):
        r[i], prob[i] = scipy.stats.pearsonr(y_true[:,i],y_pred[:,i])
    plt.plot(unpack(r,vari,axis=0), lev, label=label)
    plt.ylim([np.amax(lev),np.amin(lev)])
    plt.ylabel('$\sigma$')
    plt.title('Correlation Coefficient')
    
def plot_rmse(y_true,y_pred,vari,lev, label=None):
    rmse = np.sqrt(metrics.mean_squared_error(y_true,y_pred,multioutput='raw_values'))
    rmse = rmse / np.mean(y_true, axis=0)
    plt.plot(unpack(rmse,vari,axis=0), lev, label=label)
    plt.ylim([np.amax(lev),np.amin(lev)])
    plt.ylabel('$\sigma$')
    plt.xlabel(out_str_dict[vari])
    plt.title('Root Mean Squared Error/mean')
    
def plot_expl_var(y_true,y_pred,vari,lev, label=None):
    expl_var = metrics.explained_variance_score(y_true,y_pred,multioutput='raw_values')
    plt.plot(unpack(expl_var,vari,axis=0) ,lev, label=label)
    plt.ylim([np.amax(lev),np.amin(lev)])
    plt.ylabel('$\sigma$')
    #plt.xlabel('(' + out_str_dict[vari] + ')$^2$')
    plt.title('Explained Variance Regression Score')
    
def _plot_enthalpy(y, dlev, label=None):
    k = calc_enthalpy(y, dlev)
    n, bins, patches = plt.hist(k, 50, alpha=0.5,label=label) #, facecolor='green'
    plt.title('Heating rate needed to conserve column enthalpy')
    plt.xlabel('K/day over column')
    
def _plot_precip(y_true,y_pred, dlev):
    y_true = calc_precip(y_true, dlev)
    y_pred = calc_precip(y_pred, dlev)
    ind = y_true.argsort()
    plt.plot(y_true[ind], label='actual')
    plt.plot(y_pred[ind], alpha=0.6, label='predict')
    plt.legend(loc="upper left")
    plt.title('Precipitation Rate [mm/day]')
    plt.xlabel('Sorted by actual rate')

def check_scaling_distribution(x_unscl, x_scl, y_unscl, y_scl, lat, lev, figpath):
    # For input variables
    fig, ax = plt.subplots(2, 2)
    _plot_distribution(unpack(x_unscl,'T'), lat, lev, fig, ax[0,0], './figs/','T (unscaled) [K]','')
    _plot_distribution(unpack(x_scl,  'T'), lat, lev, fig, ax[0,1], './figs/','T (scaled) []','')
    _plot_distribution(unpack(x_unscl,'q'), lat, lev, fig, ax[1,0], './figs/','q (unscaled) [g/kg]','')
    _plot_distribution(unpack(x_scl,  'q'), lat, lev, fig, ax[1,1], './figs/','q (scaled) []','')
    fig.savefig(figpath + 'input_scaling_check.png',bbox_inches='tight',dpi=450)
    # For output variables
    fig, ax = plt.subplots(2, 2)
    _plot_distribution(unpack(y_unscl,'T'), lat, lev, fig, ax[0,0], './figs/','T tend (unscaled) [K/day]','')
    _plot_distribution(unpack(y_scl,  'T'), lat, lev, fig, ax[0,1], './figs/','T tend (scaled) []','')
    _plot_distribution(unpack(y_unscl,'q'), lat, lev, fig, ax[1,0], './figs/','q tend (unscaled) [g/kg/day]','')
    _plot_distribution(unpack(y_scl,  'q'), lat, lev, fig, ax[1,1], './figs/','q tend(scaled) []','')
    fig.savefig(figpath + 'output_scaling_check.png',bbox_inches='tight',dpi=450)

def check_output_distribution(yt_unscl, yt_scl, yp_unscl, yp_scl, lat, lev, figpath):
    # For unscaled variables
    fig, ax = plt.subplots(2, 2)
    _plot_distribution(unpack(yt_unscl,'T'), lat, lev, fig, ax[0,0], './figs/','T true (unscld) [K/day]','')
    _plot_distribution(unpack(yp_unscl,'T'), lat, lev, fig, ax[0,1], './figs/','T pred (unscld) [K/day]','')
    _plot_distribution(unpack(yt_unscl,'q'), lat, lev, fig, ax[1,0], './figs/','q true (unscld) [g/kg/day]','')
    _plot_distribution(unpack(yp_unscl,'q'), lat, lev, fig, ax[1,1], './figs/','q pred (unscld) [g/kg/day]','')
    fig.savefig(figpath + 'output_compare_true_pred_unscaled.png',bbox_inches='tight',dpi=450)
    # For scaled variables
    fig, ax = plt.subplots(2, 2)
    _plot_distribution(unpack(yt_scl,'T'), lat, lev, fig, ax[0,0], './figs/','T true (scld) []','')
    _plot_distribution(unpack(yp_scl,'T'), lat, lev, fig, ax[0,1], './figs/','T pred (scld) []','')
    _plot_distribution(unpack(yt_scl,'q'), lat, lev, fig, ax[1,0], './figs/','q true (scld) []','')
    _plot_distribution(unpack(yp_scl,'q'), lat, lev, fig, ax[1,1], './figs/','q pred (scld) []','')
    fig.savefig(figpath + 'output_compare_true_pred_scaled.png',bbox_inches='tight',dpi=450)
        

def _plot_distribution(z, lat, lev, fig, ax, figpath, titlestr, xstr):
    """Plots a stack of histograms of log10(data) at all levels"""
    # Initialize the bins and the frequency 
    num_bins = 100
    bins = np.linspace(np.amin(z), np.amax(z), num_bins+1)
    n = np.zeros((num_bins, lev.size))
    # Calculate distribution at each level 
    for i in range(lev.size):
        n[:,i], _ = np.histogram(z[:,i], bins=bins)
    bins=bins[:-1]
    # Take a logarithm and deal with case where we take log of 0
    n = np.log10(n)
    n_small = np.amin(n[np.isfinite(n)])
    n[np.isinf(n)]  = n_small
    # Plot histogram
    ca = ax.contourf(bins, lev, n.T)
    ax.set_ylim(1,0)
    plt.colorbar(ca, ax=ax)
    ax.set_xlabel(xstr)
    ax.set_ylabel(r'$\sigma$')
    ax.set_title(titlestr)

def plot_model_error_over_time(errors, mlp_str, fig_dir):
    x = np.arange(errors.shape[0])
    ytix = [.5e-3,1e-3,2e-3,5e-3,10e-3,20e-3]
    # Plot error rate vs. iteration number
    fig=plt.figure()
    # Plot training errors
    plt.semilogy(x, np.squeeze(errors[:,0]), alpha=0.5,color='blue',label='Training')
    plt.semilogy(x, np.squeeze(errors[:,1]), alpha=0.5,color='blue')
    plt.yticks(ytix,ytix)
    plt.ylim((np.nanmin(errors), np.nanmax(errors)))
    plt.semilogy(x, np.squeeze(errors[:,2]), alpha=0.5,label='Testing',color='green')
    plt.semilogy(x, np.squeeze(errors[:,3]), alpha=0.5,color='green')
    plt.legend()
    plt.title('Error for ' + mlp_str)
    plt.xlabel('Iteration Number')
    fig.savefig(fig_dir + 'error_history.png',bbox_inches='tight',dpi=450)

# ----  HELPER SCRIPTS  ---- # 
    
def calc_enthalpy(y, dlev):
    # y is output data set in rate (1/day)
    # k is the implied uniform heating rate over the whole column to correct the imbalance
    cp = 1005. #J/kg/K
    L = 2.5e6 #J/kg
    k = (unpack(y, 'T') + (L/cp) * unpack(y, 'q')/1000.)
    k = vertical_integral(k, dlev)
    k = k / 1e5
    return k

def vertical_integral(data, dlev):
    g = 9.8 #m/s2
    data = -1./g * np.sum(data * dlev[:,None].T, axis=1)*1e5
    return data
  
def calc_precip(y, dlev):
    y = unpack(y,'q')
    y = y / 1000. # kg/kg/day
    return vertical_integral(y, dlev) #mm/day
