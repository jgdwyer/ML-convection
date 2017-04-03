import numpy as np
import sknn_jgd.mlp
import time
from sklearn.ensemble import RandomForestRegressor
import src.nnload as nnload
import pickle
import src.nnplot as nnplot
import os

# ---  BUILDING NEURAL NETS  --- #
def TrainNNwrapper(num_layers, hidneur, x_ppi, y_ppi,
                     n_iter=None, n_stable=None,
                     minlev=0.0, weight_precip=False, weight_shallow=False,
                     weight_decay=0.0, rainonly=False, noshallow=False,
                     N_trn_exs=None, convcond=False, doRF=False,
                     cirrusflag=False, plot_training_results=False):
    """Loads training data and trains and stores neural network

    Args:
        num_layers (int): Number of layers in the NN
        hidneur (int): Number of hidden neurons in each layer
        x_ppi (dict): The type of preprocessing to do to the features (inputs)
        y_ppi (dict): The type of preprocessing to do to the targets (outputs)
        n_iter (int): Number of iterations
        n_stable (int): Number of iterations after stability reached
        minlev (float): Don't train on data above this sigma level (assume 0)
        weight_precip (bool): Weight training examples by precipitation amount
        weight_shallow (bool): Weight training examples by shallow convection
        weight_decay (float): Regularization strength. 0 is no regularization
        rainonly (bool): Only train on precipitating examples
        noshallow (bool): Don't train on shallow convective events
        N_trn_exs (int): Number of training examples to learn on
        convcond (bool): If true, learn to do convection + condensation
        doRF (bool): Use a random forest rather than an ANN
        cirrusflag (bool): Run on the cirrus machine
        plot_training_results (bool): Whether to also plot the model on training data
    Returns:
        str: String id of trained NN
    """
    # Loads data
    datadir, trainfile, testfile, pp_str = nnload.GetDataPath(cirrusflag, convcond)
    x, y, cv, Pout, lat, lev, dlev, timestep = nnload.LoadData(trainfile, minlev, rainonly=rainonly,
                                                               noshallow=noshallow, N_trn_exs=N_trn_exs)
    # Prepare data
    w = TrainingWeights(y, Pout, lev, weight_precip, weight_shallow)
    x_pp, x, y_pp, y, pp_str = PreprocessData(x_ppi, x, y_ppi, y, pp_str, N_trn_exs)
    regularize = CatchRegularization(weight_decay)
    # Either build a random forest or build a neural netowrk
    if doRF:
        r_mlp, r_str = BuildRandomForest(500, pp_str)
        errors_stored = np.ones((1, 6))
    else:
        r_mlp, r_str = BuildNN('regress', num_layers, 'Rectifier', hidneur,
                                'momentum', pp_str, batch_size=100,
                                n_stable=n_stable, n_iter=n_iter,
                                learning_momentum=0.9, learning_rate=0.01,
                                regularize=regularize,
                                weight_decay=weight_decay,
                                valid_size=0.2)
    r_str = UpdateMLPname(weight_precip, weight_shallow, r_str)
    # Print details about the ML algorithm we are using
    print(r_str + ' Using ' + str(x.shape[0]) + ' training examples with ' +
          str(x.shape[1]) + ' input features and ' + str(y.shape[1]) +
          ' output targets')
    # Train the neural network
    r_mlp, r_errors = TrainNN(r_mlp, r_str, x, y, w)
    # Save the neural network to access it later
    SaveNN(r_mlp, r_str, r_errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, dlev)
    # Plot figures with validation data (and with training data)
    nnplot.PlotAllFigs(r_str, testfile, noshallow=noshallow,
                         rainonly=rainonly)
    if plot_training_results:
        nnplot.PlotAllFigs(r_str, trainfile, validation=False,
                             noshallow=noshallow, rainonly=rainonly)
    return r_str


def TrainingWeights(y, Pout, lev, weight_precip, weight_shallow):
    """Set up weights for training examples"""
    wp = np.ones(y.shape[0])
    ws = np.ones(y.shape[0])
    if weight_precip:
        wp = Pout + 1
    if weight_shallow:
        # 0.8 is near the level of maximum shallow convective activity
        shallow_lev = np.argmin(np.abs(lev - 0.8))
        q = nnload.unpack(y, 'q')
        # Find where moistening is larger than some threshold
        ind = np.argwhere(q[:, shallow_lev] >= 5)
        # Set threshold events as proportional
        ws[ind] = q[ind, shallow_lev]
    # Combine weights
    w = wp * ws
    # Or set weights to none
    if (not (weight_precip) and not (weight_shallow)):
        w = None
    return w

def PreprocessData(x_ppi, x, y_ppi, y, pp_str, N_trn_exs):
    """Transform data according to input preprocessor requirements and make
    make preprocessor string for saving"""
    x_pp = nnload.init_pp(x_ppi, x)
    x = nnload.transform_data(x_ppi, x_pp, x)
    y_pp = nnload.init_pp(y_ppi, y)
    y = nnload.transform_data(y_ppi, y_pp, y)
    # Make preprocessor string for saving
    pp_str = pp_str + 'X-' + x_ppi['name'] + '-' + x_ppi['method'][:6] + '_'
    pp_str = pp_str + 'Y-' + y_ppi['name'] + '-' + y_ppi['method'][:6] + '_'
    # Add number of training examples to string
    pp_str = pp_str + 'Ntrnex' + str(N_trn_exs) + '_'
    return x_pp, x, y_pp, y, pp_str

def CatchRegularization(weight_decay):
    """scikit-neuralnetwork seems to have a bug if regularization is set to zero"""
    if weight_decay > 0.0:
        regularize = 'L2'
    else:
        regularize = None
    return regularize

def UpdateMLPname(weight_precip, weight_shallow, r_str):
    if weight_precip:
        r_str = r_str + '_wpr'
    if weight_shallow:
        r_str = r_str + '_wsh'
    r_str = r_str + '_v3'  # reflects that we are loading v3 of training data

def SaveNN(r_mlp, r_str, r_errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, dlev):
    """Save neural network"""
    if not os.path.exists('./data/regressors/'):
        os.makedirs('./data/regressors/')
    pickle.dump([r_mlp, r_str, r_errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev,
                 dlev], open('./data/regressors/' + r_str + '.pkl', 'wb'))

def store_stats(i, avg_train_error, best_train_error, avg_valid_error,
                best_valid_error, avg_train_obj_error, best_train_obj_error,
                **_):
    if i == 1:
        global errors_stored
        errors_stored = []
    errors_stored.append((avg_train_error, best_train_error,
                          avg_valid_error, best_valid_error,
                          avg_train_obj_error, best_train_obj_error))


def BuildNN(method, num_layers, actv_fnc, hid_neur, learning_rule, pp_str,
             batch_size=100, n_iter=None, n_stable=None,
             learning_rate=0.01, learning_momentum=0.9,
             regularize='L2', weight_decay=0.0, valid_size=0.5,
             f_stable=.001):
    """Builds a multi-layer perceptron via the scikit neural network interface
    """
    # First build layers
    actv_fnc = num_layers*[actv_fnc]
    hid_neur = num_layers*[hid_neur]
    layers = [sknn_jgd.mlp.Layer(f, units=h) for f, h in zip(actv_fnc,
                                                             hid_neur)]
    # Append a linear output layer if regressing and a softmax layer if
    # classifying
    if method == 'regress':
        layers.append(sknn_jgd.mlp.Layer("Linear"))
        mlp = sknn_jgd.mlp.Regressor(layers,
                                     n_iter=n_iter,
                                     batch_size=batch_size,
                                     learning_rule=learning_rule,
                                     learning_rate=learning_rate,
                                     learning_momentum=learning_momentum,
                                     regularize=regularize,
                                     weight_decay=weight_decay,
                                     n_stable=n_stable,
                                     valid_size=valid_size,
                                     f_stable=f_stable,
                                     callback={'on_epoch_finish': store_stats})
    if method == 'classify':
        layers.append(sknn_jgd.mlp.Layer("Softmax"))
        mlp = sknn_jgd.mlp.Classifier(layers,
                                      n_iter=n_iter,
                                      batch_size=batch_size,
                                      learning_rule=learning_rule,
                                      learning_rate=learning_rate,
                                      learning_momentum=learning_momentum,
                                      regularize=regularize,
                                      weight_decay=weight_decay,
                                      n_stable=n_stable,
                                      valid_size=valid_size,
                                      callback={'on_epoch_finish':
                                                store_stats})
    # Write nn string
    layerstr = '_'.join([str(h) + f[0] for h, f in zip(hid_neur, actv_fnc)])
    if learning_rule == 'momentum':
        lrn_str = str(learning_momentum)
    else:
        lrn_str = str(learning_rate)
    # Construct name
    mlp_str = pp_str + method[0] + "_" + layerstr + "_" + learning_rule[0:3] +\
        lrn_str
    # If using regularization, add that to the name too
    if weight_decay > 0.0:
        mlp_str = mlp_str + 'reg' + str(weight_decay)
    # Add the number of iterations too
    mlp_str = mlp_str + '_Niter' + str(n_iter)
    return mlp, mlp_str


def BuildRandomForest(N_trees, mlp_str):
    mlp = RandomForestRegressor(n_estimators=N_trees)
    mlp_str = 'RF_regress_' + str(N_trees)
    return mlp, mlp_str


def TrainNN(mlp, mlp_str, x, y, w=None):
    """Train each item in a list of multi-layer perceptrons and then score
    on test data. Expects that mlp is a list of MLP objects"""
    # Initialize
    start = time.time()
    # Train the model using training data
    mlp.fit(x, y, w)
    train_score = mlp.score(x, y)
    end = time.time()
    print("Training Score: {:.4f} for Model {:s} ({:.1f} seconds)".format(
                                              train_score, mlp_str, end-start))
    # This is an N_iter x 4 array...see score_stats
    errors = np.asarray(errors_stored)
    # Return the fitted models and the scores
    return mlp, errors


def classify(classifier, x, y):
    """Applies a trained classifier to input x and y and
    outputs data when classifier expects convection is occurring."""
    # Apply classifier
    out = classifier.predict(x)
    # Get samples of when classifier predicts that it is convecting
    ind = np.squeeze(np.not_equal(out, 0))
    # Store data only when it is convection is predicted
    x = x[ind, :]
    y = y[ind, :]
    return x, y
