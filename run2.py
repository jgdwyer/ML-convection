import src.nntrain as nntrain
import src.nnplot as nnplot
import src.nnload as nnload

# Define preprocessor
x_ppi={'name':'StandardScaler','method':'qTindividually'}
y_ppi={'name':'SimpleY','method':'qTindividually'}
#y_ppi={'name':'MaxAbs','method':'qTindividually'}

n_samps = [100,200,500,1000,2000,3500,5000,7500,10000,12500,15000]

#hid_neur = [10,20,40,60,100,150,225,300,450,600,800,1000]
# hid_neur = [5,15,30,50,80]
# nntrain.train_nn_wrapper(2, 61, x_ppi, y_ppi, n_iter=10000, minlev=0.25, noshallow=True)
for n_samp in n_samps:
    nntrain.train_nn_wrapper(2, 60, x_ppi, y_ppi, minlev=0.25, n_iter=10000,
                            rainonly=False,  use_weights=False, weight_decay=0.0,
                            N_trn_exs=n_samp)
# for h in hid_neur:
#     r_str = nntrain.train_nn_wrapper(1, h, x_ppi, y_ppi, n_iter=10000, minlev=0.25,
#      use_weights=False, weight_decay=0.0)
#     r_str = nntrain.train_nn_wrapper(1, h, x_ppi, y_ppi, n_iter=10000, minlev=0.25,
#      use_weights=False, weight_decay=0.00001)
#     r_str = nntrain.train_nn_wrapper(2, h, x_ppi, y_ppi, n_iter=10000, minlev=0.25,
#      use_weights=False, weight_decay=0.0)
#     r_str = nntrain.train_nn_wrapper(2, h, x_ppi, y_ppi, n_iter=10000, minlev=0.25,
#      use_weights=False, weight_decay=0.00001)
#r_str[0] = nntrain.train_nn_wrapper(4, 200, x_ppi, y_ppi, n_iter=5000, minlev=0.25, use_weights=True, weight_decay=0.00001)
#r_str[0] = nntrain.train_nn_wrapper(4, 1000, x_ppi, y_ppi, n_iter=4000, minlev=0.25, use_weights=True, weight_decay=0.0)
#r_str[1] = nntrain.train_nn_wrapper(50, x_ppi, y_ppi, n_iter=1000, minlev=0.25, use_weights=True, weight_decay=0.00001)
#r_str[2] = nntrain.train_nn_wrapper(50, x_ppi, y_ppi, n_iter=1000, minlev=0.25, use_weights=True, weight_decay=0.0001)
#r_str[3] = nntrain.train_nn_wrapper(50, x_ppi, y_ppi, n_iter=1000, minlev=0.25, use_weights=True, weight_decay=0.001)
#r_str[4] = nntrain.train_nn_wrapper(50, x_ppi, y_ppi, n_iter=1000, minlev=0.25, use_weights=True, weight_decay=0.01)
#r_str[5] = nntrain.train_nn_wrapper(50, x_ppi, y_ppi, n_iter=1000, minlev=0.25, use_weights=True, weight_decay=0.1)
#r_str[6] = nntrain.train_nn_wrapper(50, x_ppi, y_ppi, n_iter=1000, minlev=0.25, use_weights=True, weight_decay=1.0)
#r_str[7] = nntrain.train_nn_wrapper(50, x_ppi, y_ppi, n_iter=1000, minlev=0.25, use_weights=True, weight_decay=10.0)
#r_str[8] = nntrain.train_nn_wrapper(50, x_ppi, y_ppi, n_iter=1000, minlev=0.25, use_weights=True, weight_decay=100.0)
#r_str[1], x, y = nntrain.train_nn_wrapper(800, x_ppi, y_ppi, n_stable=100, minlev=0.25)
#r_str[2], x, y = nntrain.train_nn_wrapper(1200, x_ppi, y_ppi, n_stable=100)
#r_str[3], x, y = nntrain.train_nn_wrapper(1600, x_ppi, y_ppi, n_stable=100)

#nnplot.plot_all_figs('X-StandardScaler-indivi_Y-MaxAbs-qTindi_r_50R_mom0.5reg0.001_w2')
