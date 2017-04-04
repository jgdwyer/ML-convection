import src.nntrain as nntrain
import src.nnplot as nnplot
import src.nnload as nnload
import numpy as np

# Define preprocessor
x_ppi = {'name': 'StandardScaler', 'method': 'qTindividually'}
y_ppi = {'name': 'SimpleY', 'method': 'qTindividually'}

# Define other arguments
num_layers = 1
N_neurons = 60

# Run an example with many neurons in first layer, but not many in second layer
nntrain.TrainNNwrapper(num_layers, N_neurons, x_ppi, y_ppi, minlev=0.2,
                 n_iter=1000, rainonly=False, weight_decay=1e-6,
                 N_trn_exs=100, weight_precip=False,
                 weight_shallow=False, convcond=False,
                 cirrusflag=False, plot_training_results=False)

