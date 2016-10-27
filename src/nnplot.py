import numpy as np
import matplotlib
matplotlib.use('Agg') # so figs just print to file
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import src.nnload as nnload
import scipy.stats 
from sklearn import metrics, preprocessing
import pickle
import os
unpack = nnload.unpack

# ---   META PLOTTING SCRIPTS  --- #
#def plots_by_lat(pp_x, pp_y, r_mlp, lat):

def plot_all_figs(r_str, datasource='./data/convection_50day_validation.pkl', validation=True):
    # Open the neural network and the preprocessing scheme
    r_mlp_eval, _, errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, dlev = \
           pickle.load(open('./data/regressors/' + r_str + '.pkl', 'rb'))
    # Open the validation data set
    x_unscl, ytrue_unscl,_,_,_,_,_,_ = nnload.loaddata(datasource, minlev=min(lev))
    # Set figure path and create directory if it does not exist
    figpath = './figs/' + r_str + '/'
    # If plotting on training data create a new subfolder
    if validation is False:
        figpath = figpath + 'training_data/'
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    # Scale data using input scalers
    x_scl     = nnload.transform_data(x_ppi, x_pp, x_unscl)
    ytrue_scl = nnload.transform_data(y_ppi, y_pp, ytrue_unscl)
    # Apply neural network to get predicted output
    ypred_scl   = r_mlp_eval.predict(x_scl)
    ypred_unscl = nnload.inverse_transform_data(y_ppi, y_pp, ypred_scl)
    # Do plotting
    # Plot model errors over iteration history
#    plot_model_error_over_time(errors, r_str, figpath)
    # Plot historgram showing how scaling changed character of input and output data
#    check_scaling_distribution(x_unscl, x_scl, ytrue_unscl, ytrue_scl, lat, lev, figpath)
    # Plot histogram showing how well true and predicted values match
#    check_output_distribution(ytrue_unscl, ytrue_scl, ypred_unscl, ypred_scl, lat, lev, figpath)
    # Plot means and standard deviations
#    plot_means_stds(ytrue_unscl, ypred_unscl, lev, figpath)
    # Plot correlation coefficient, explained variance, and rmse
#    plot_error_stats(ytrue_unscl, ypred_unscl, lev, figpath)
    # Plot a "time series" of precipitaiton
#    plot_precip(ytrue_unscl, ypred_unscl, dlev, figpath)
    # Plot a scatter plot of true vs predicted precip
    plot_scatter(ytrue_unscl, ypred_unscl, lev, dlev, figpath)
    # Plot the enthalpy conservation
#    plot_enthalpy(ytrue_unscl, ypred_unscl, dlev, figpath)
    # Plot some example profiles
#    plot_sample_profiles(20, x_unscl, ytrue_unscl, ypred_unscl, lev, figpath)
    # Plot mean, bias, rmse, r^2  (lat vs lev)
#    make_contour_plots(figpath, x_ppi, y_ppi, x_pp, y_pp, r_mlp_eval, lat, lev)

def make_contour_plots(figpath, x_ppi, y_ppi, x_pp, y_pp, r_mlp_eval, lat, lev):
    Tmean, qmean, Tbias, qbias, rmseT, rmseq, rT, rq = nnload.stats_by_latlev(x_ppi, y_ppi, x_pp, y_pp, r_mlp_eval, lat, lev)
    # Make figs
    # True means
    f,ax1,ax2 = plot_contour(Tmean,qmean,lat,lev, avg_hem=False)
    ax1.set_title('Temperature True Mean [K/day]')
    ax2.set_title('Humidity True Mean [kg/kg/day]')
    f.savefig(figpath + 'latlev_truemean.png', bbox_inches='tight',dpi=450)
    plt.close()
    # Bias from true mean
    f,ax1,ax2 = plot_contour(Tbias,qbias,lat,lev, avg_hem=False)
    ax1.set_title('Temperature Mean Bias [K/day]')
    ax2.set_title('Humidity Mean Bias [kg/kg/day]')
    f.savefig(figpath + 'latlev_bias.png', bbox_inches='tight',dpi=450)
    plt.close()
    # Root mean squared error
    f,ax1,ax2 = plot_contour(rmseT,rmseq,lat,lev, avg_hem=False)
    ax1.set_title('Temperature RMSE [K/day]')
    ax2.set_title('Humidity RMSE [kg/kg/day]')
    f.savefig(figpath + 'latlev_rmse.png', bbox_inches='tight',dpi=450)
    plt.close()
    # Pearson r Correlation Coefficient
    f,ax1,ax2 = plot_contour(rT, rq, lat, lev, avg_hem=False)
    ax1.set_title('Temperature CorrelationCoefficient')
    ax2.set_title('Humidity Correlation Coefficient')
    f.savefig(figpath + 'latlev_corrcoeff.png', bbox_inches='tight',dpi=450)
    plt.close()

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
    plt.close()
# Plot means and standard deviations
def plot_means_stds(y3_true, y3_pred, lev, figpath):
    fig = plt.figure()
    do_mean_or_std('mean','T',y3_true,y3_pred, lev, 1)
    do_mean_or_std('mean','q',y3_true,y3_pred, lev, 2)
    do_mean_or_std('std','T',y3_true,y3_pred, lev, 3)
    do_mean_or_std('std','q',y3_true,y3_pred, lev, 4)
    fig.savefig(figpath + 'regress_means_stds.png', bbox_inches='tight',dpi=450)
    plt.close()

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
    plt.close()

#Plot a time series of precipitaiton
def plot_precip(y3_true, y3_pred, dlev, figpath):
    fig = plt.figure()
    _plot_precip(y3_true,y3_pred, dlev)
    fig.savefig(figpath + 'regress_P_rate.png',bbox_inches='tight',dpi=450)
    plt.close()

# Plot a scatter plot of true vs predicted for some variable
def plot_scatter(ytrue_unscl, ypred_unscl, lev, dlev, figpath):
    # Plot scatter of precipitation
    P_true = calc_precip(ytrue_unscl, dlev)
    P_pred = calc_precip(ypred_unscl, dlev)
    f = plt.figure()
    _plot_scatter(plt.gca(), P_true, P_pred, titstr='Precipitation Rate [mm/day]')
    Plessthan0 = sum(P_pred < 0.0)
    Plessthan0pct = 100.*Plessthan0/len(P_pred)
    plt.text(0.01,0.95,"Pred. P<0 {:.1f}% of time".format(Plessthan0pct),
             transform=plt.gca().transAxes)
    # JGD TO DO: ADD BEST FIT LINE
    f.savefig(figpath + 'P_scatter.png',bbox_inches='tight',dpi=450)
    plt.close()
    # Plot scatters at each level
    # First create new folder
    if not os.path.exists(figpath + '/scatters/'):
        os.makedirs(figpath + '/scatters/')
    for i in range(np.size(lev)):
        f, ax = plt.subplots(1, 2)
        Ttrue = unpack(ytrue_unscl, 'T')[:,i]
        Tpred = unpack(ypred_unscl, 'T')[:,i]
        qtrue = unpack(ytrue_unscl, 'q')[:,i]
        qpred = unpack(ypred_unscl, 'q')[:,i]
        lev_str = r'$\sigma$ = {:.2f}'.format(lev[i])
        _plot_scatter(ax[0], Ttrue, Tpred, titstr='T [K/day] at '+lev_str)
        _plot_scatter(ax[1], qtrue, qpred, titstr='q [g/kg/day] at '+lev_str)
        Teq0 = sum(Ttrue==0.0) / len(Ttrue) * 100.
        qeq0 = sum(qtrue==0.0) / len(qtrue) * 100.
        ax[0].text(0.01, 0.95, 'True T=0 {:.1f}% of time'.format(Teq0),
                   transform=ax[0].transAxes)
        ax[1].text(0.01, 0.95, 'True q=0 {:.1f}% of time'.format(qeq0),
                   transform=ax[1].transAxes)
        f.savefig(figpath + '/scatters/Tq_scatter_sigma{:.2f}.png'.format(lev[i]),
                  bbox_inches='tight', dpi=450)
        plt.close()

def _plot_scatter(ax, true, pred, titstr=None):
    ax.scatter(true, pred, s=5, alpha=0.25)
    xmin = np.min(true)
    xmax = np.max(true)
    ymin = np.min(pred)
    ymax = np.max(pred)
    xymin = np.min([xmin,ymin])
    xymax = np.max([xmax,ymax])
    ax.plot([xymin,xymax], [xymin, xymax], color='k', ls='--')
    ax.set_xlim(xymin, xymax)
    ax.set_ylim(xymin, xymax)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    if titstr is not None:
        ax.set_title(titstr)

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
    plt.close()

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
    plt.close()
    # For output variables
    fig, ax = plt.subplots(2, 2)
    _plot_distribution(unpack(y_unscl,'T'), lat, lev, fig, ax[0,0], './figs/','T tend (unscaled) [K/day]','')
    _plot_distribution(unpack(y_scl,  'T'), lat, lev, fig, ax[0,1], './figs/','T tend (scaled) []','')
    _plot_distribution(unpack(y_unscl,'q'), lat, lev, fig, ax[1,0], './figs/','q tend (unscaled) [g/kg/day]','')
    _plot_distribution(unpack(y_scl,  'q'), lat, lev, fig, ax[1,1], './figs/','q tend(scaled) []','')
    fig.savefig(figpath + 'output_scaling_check.png',bbox_inches='tight',dpi=450)
    plt.close()

def check_output_distribution(yt_unscl, yt_scl, yp_unscl, yp_scl, lat, lev, figpath):
    # For unscaled variables
    fig, ax = plt.subplots(2, 2)
    x1, x2, bins = _plot_distribution(unpack(yp_unscl,'T'), lat, lev, fig, ax[0,1], './figs/','T pred (unscld) [K/day]','')
    _plot_distribution(unpack(yt_unscl,'T'), lat, lev, fig, ax[0,0], './figs/','T true (unscld) [K/day]','',x1,x2, bins)
    x1,x2, bins=_plot_distribution(unpack(yp_unscl,'q'), lat, lev, fig, ax[1,1], './figs/','q pred (unscld) [g/kg/day]','')
    _plot_distribution(unpack(yt_unscl,'q'), lat, lev, fig, ax[1,0], './figs/','q true (unscld) [g/kg/day]','',x1,x2, bins)
    fig.savefig(figpath + 'output_compare_true_pred_unscaled.png',bbox_inches='tight',dpi=450)
    plt.close()
    # For scaled variables
    fig, ax = plt.subplots(2, 2)
    x1,x2,bins=_plot_distribution(unpack(yp_scl,'T'), lat, lev, fig, ax[0,1], './figs/','T pred (scld) []','')
    _plot_distribution(unpack(yt_scl,'T'), lat, lev, fig, ax[0,0], './figs/','T true (scld) []','',x1,x2, bins)
    x1,x2,bins=_plot_distribution(unpack(yp_scl,'q'), lat, lev, fig, ax[1,1], './figs/','q pred (scld) []','')
    _plot_distribution(unpack(yt_scl,'q'), lat, lev, fig, ax[1,0], './figs/','q true (scld) []','',x1,x2, bins)
    fig.savefig(figpath + 'output_compare_true_pred_scaled.png',bbox_inches='tight',dpi=450)
    plt.close()

def _plot_distribution(z, lat, lev, fig, ax, figpath, titlestr, xstr,xl=None,xu=None, bins=None):
    """Plots a stack of histograms of log10(data) at all levels"""
    # Initialize the bins and the frequency 
    num_bins = 100
    if bins is None:
        bins = np.linspace(np.amin(z), np.amax(z), num_bins+1)
    n = np.zeros((num_bins, lev.size))
    # Calculate distribution at each level 
    for i in range(lev.size):
        n[:,i], _ = np.histogram(z[:,i], bins=bins)
    bins1=bins[:-1]
    # Take a logarithm and deal with case where we take log of 0
    n = np.log10(n)
    n_small = np.amin(n[np.isfinite(n)])
    n[np.isinf(n)]  = n_small
    # Plot histogram
    ca = ax.contourf(bins[:-1], lev, n.T)
    ax.set_ylim(1,0)
    if xl is not None:
        ax.set_xlim(xl,xu)
    plt.colorbar(ca, ax=ax)
    ax.set_xlabel(xstr)
    ax.set_ylabel(r'$\sigma$')
    ax.set_title(titlestr)
    xl,xr = ax.set_xlim()
    return xl, xr, bins

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
    plt.close()

def plot_regressors_scores(r_list,r_str,x_test,y_true, fig_dir, txt):
    """Given a list of fitted regressor objects, compare their skill on a variety of tests"""
    mse=[]
    r2_u=[]
    r2_w=[]
    exp_var_u=[]
    exp_var_w=[]
    for reg in r_list:
        y_pred = reg.predict(x_test)
        mse.append(metrics.mean_squared_error(y_true,y_pred,multioutput='uniform_average'))
        r2_u.append(metrics.r2_score(y_true,y_pred,multioutput='uniform_average'  ))
        r2_w.append(metrics.r2_score(y_true,y_pred,multioutput='variance_weighted'))
        exp_var_u.append(metrics.explained_variance_score(y_true,y_pred,
                                                          multioutput='uniform_average'  ))
        exp_var_w.append(metrics.explained_variance_score(y_true,y_pred,
                                                          multioutput='variance_weighted'))
    fig=plt.figure()
    plt.subplot(1,2,1)
    tick=range(len(mse))
    # Plot mean squared error
    plt.yticks(tick, r_str)
    plt.semilogx(mse, tick, marker='o',)
    plt.title('Mean Squared Error')
    # Plot R2 
    plt.subplot(1,2,2)
    plt.plot(r2_u, tick, marker='o', label='uniform')
    plt.plot(r2_w, tick, marker='o', label='weighted')
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.legend(loc="upper left")
    plt.title('R^2 score')
    plt.xlim((-1,1))
    fig.savefig(fig_dir + txt + '_scores.png',bbox_inches='tight',dpi=450)
    plt.close()
 
def plot_sample_profiles(num_prof, x, ytrue, ypred, lev, figpath):
    # Make directory if one does not exist
    if not os.path.exists(figpath + '/samples/'):
        os.makedirs(figpath + '/samples/')
    for i in range(num_prof):
        samp = np.random.randint(0, x.shape[0])
        plot_sample_profile(x[samp,:], ytrue[samp,:], ypred[samp,:], 
                             lev, filename=figpath+'/samples/'+str(samp)+'.png')

def plot_sample_profile(x, y_true, y_pred, lev, filename=None):
    """Plots the vertical profiles of input T & q and predicted and true output tendencies"""
    f = plt.figure()
    gs = gridspec.GridSpec(1, 4)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2:])
    #f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    T = nnload.unpack(x, 'T', axis=0)
    q = nnload.unpack(x, 'q', axis=0)
    theta = calc_theta(T, lev)
    # Plot input temperature profile
    ax1.plot(theta, lev)
    ax1.set_ylim(1, 0.25)
    ax1.set_xlim(280, 350)
    ax1.set_title(r'Input Temp (as $\theta$)')
    ax1.set_xlabel(r'$\theta$ [K]')
    ax1.grid(True)
    # Plot input humidity profile
    ax2.plot(q, lev)
    ax2.set_ylim(1, 0.25)
    ax2.set_xlim(0, .02)
    ax2.set_title('Input Humidity')
    ax2.set_xlabel('q [g/kg]')
    ax2.grid(True)
    # Plot output profiles
    ax3.plot(nnload.unpack(y_true, 'T', axis=0), lev, color='red' , ls='-' , label='T true')
    ax3.plot(nnload.unpack(y_pred, 'T', axis=0), lev, color='red' , ls='--', label='T pred')
    ax3.plot(nnload.unpack(y_true, 'q', axis=0), lev, color='blue', ls='-' , label='q true')
    ax3.plot(nnload.unpack(y_pred, 'q', axis=0), lev, color='blue', ls='--', label='q pred')
    ax3.set_ylim(1, 0.25)
    ax3.set_xlabel('T [K/day] or q [g/kg/day] Tend')
    ax3.set_title('Output Temp and Humidity')
    ax3.legend()
    ax3.grid(True)
    # Save file if requested
    if filename is not None:
        f.savefig(filename, bbox_inches='tight', dpi=450)
    plt.close()
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

def calc_theta(T, sigma):
    kappa = 287./1005.
    theta = T * np.power(1. / sigma, kappa)
    return theta
