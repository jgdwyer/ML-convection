
# coding: utf-8

# ## Import and preprocess datasets
import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier
import os
from sklearn import preprocessing, metrics
from importlib import reload
import sknn.mlp
from sklearn import tree  #for graphing random forest tree
import pickle
pylab.rcParams['figure.figsize'] = (10, 6)
inline_rc = dict(mpl.rcParams)
import src.nnload as nnload
import src.nntrain as nntrain
import src.nnplot as nnplot

# Set script parameters
minlev = 0.0
rainonly = False
write_nn_to_netcdf = False
fig_dir = './figs/'
data_dir = './data/'


# In[2]:

# Load data
reload(nnload)
x, y, cv, Pout, lat, lev, dlev, timestep = nnload.loaddata(data_dir + 'nntest.nc', minlev,
                                                       rainonly=rainonly) #,all_lats=False,indlat=8)
# Preprocess data
scaler_x, scaler_y, x1, x2, x3, y1, y2, y3, cv1, cv2, cv3 = nnload.pp(x, y, cv, 30000)


# Train random forest
n_estimators=15
model = RandomForestRegressor(n_estimators=n_estimators)
model.fit(x1,y1)
out_train = model.predict(x1)
out_test = model.predict(x2)
# Plot feature importance
plt.plot(model.feature_importances_[0:30],lev,label="T")
plt.plot(model.feature_importances_[30:60],lev,label="q")
plt.legend()
plt.show()


# Set figure path and create directory if it does not exist
figpath = fig_dir + 'RF' + str(n_estimators) + '/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
    

# Inverse transform output back to physical units
y3_true = scaler_y.inverse_transform(y2)
y3_pred = scaler_y.inverse_transform(model.predict(x2))

# Plot means and standard deviations
nnplot.plot_means_stds(y3_true, y3_pred, lev, figpath)

# Plot correlation coefficient, explained variance, and rmse
nnplot.plot_error_stats(y3_true, y3_pred, lev, figpath)

# Plot a time series of precipitaiton
nnplot.plot_precip(y3_true, y3_pred, dlev, figpath)

# Plot the enthalpy conservation
nnplot.plot_enthalpy(y3_true, y3_pred, dlev, figpath)



