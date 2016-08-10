import numpy as np
import matplotlib.pyplot as plt
import src.nnload as nnload
import scipy.stats 
from sklearn import metrics, preprocessing
unpack = nnload.unpack

# ---   META PLOTTING SCRIPTS  --- #

# Plot rmse vs lat
def plot_rmse_vs_lat(r_mlp, figpath, data_dir='./data/', minlev=0.0, rainonly=False):
    rmseT_nh, rmseT_sh, rmseq_nh, rmseq_sh, lat = _plot_rmse_vs_lat(r_mlp,data_dir='./data/',minlev=0.0,rainonly=False)
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
    plt.show()
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
    plt.show()
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
    plt.show()
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
    
def _plot_rmse_vs_lat(r_mlp,data_dir='./data/',minlev=0.0,rainonly=False):
    # Load data
    rmseT_all = np.zeros(64)
    rmseq_all = np.zeros(64)
    for i in range(64):
        x, y, cv, Pout, lat, lev, dlev, timestep = nnload.loaddata(data_dir + 'nntest.nc', minlev,
                                                           rainonly=rainonly ,all_lats=False,indlat=i,
                                                            verbose=False) 
        # Normalize input using the scikit-learn preprocessing tools
        scaler_x = preprocessing.MinMaxScaler(feature_range=(-1.0,1.0))
        scaler_y = preprocessing.MaxAbsScaler()
        # Randomly choose samples
        samples = np.random.choice(x.shape[0], x.shape[0], replace=False)
        x3,_,_    = nnload.pp(x,  samples, scaler_x)
        y3,_,_    = nnload.pp(y,  samples, scaler_y)
        y3_pred=r_mlp.predict(x3)
        rmseT_all[i] = np.sum(np.sqrt(metrics.mean_squared_error(unpack(y3,'T'), unpack(y3_pred,'T'), 
                       multioutput='raw_values')))
        rmseq_all[i] = np.sum(np.sqrt(metrics.mean_squared_error(unpack(y3,'q'), unpack(y3_pred,'q'), 
                       multioutput='raw_values')))
        #rmse = rmse / np.mean(y3, axis=0)
    # Separate hemispheres
    rmseT_nh, rmseT_sh, lathalf = nnload.avg_hem(rmseT_all, lat, 0, split=True)
    rmseq_nh, rmseq_sh, lathalf = nnload.avg_hem(rmseq_all, lat, 0, split=True)
    return rmseT_nh.T, rmseT_sh.T, rmseq_nh.T, rmseq_sh.T, lathalf

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
    
def input_hist(x,y,vari):
    """Plot histograms of input and output data at each level (uses scaled inputs)"""
    plt.figure(figsize=(8,40))
    _,ax = plt.subplots(lev.size,2,sharex=True)
    for i in range(lev.size):
        step=.05
        bins=np.arange(-1,1+step,step)
        n,bins,_   = ax[i,0].hist(unpack(x,vari)[:,i],bins=bins,facecolor='yellow',alpha=0.5,normed=True)
        n2,bins2,_ = ax[i,1].hist(unpack(y,vari)[:,i],bins=bins,facecolor='blue'  ,alpha=0.5,normed=True)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,1].get_yaxis().set_visible(False)
        ax[i,0].set_xlim((-1,1))
        ax[i,0].set_xlim((-1,1))
        ax[i,0].set_ylim(0,np.amax(n))
        ax[i,1].set_ylim(0,np.amax(n2))
    plt.show()


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
