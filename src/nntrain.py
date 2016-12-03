import numpy as np
import sknn_jgd.mlp
import time
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import src.nnload as nnload
import pickle
import src.nnplot as nnplot


# ---  BUILDING NEURAL NETS  --- #
def train_nn_wrapper(num_layers, hidneur, x_ppi, y_ppi,
                     n_iter=None, n_stable=None,
                     minlev=0.0, weight_precip=False, weight_shallow=False,
                     weight_decay=0.0, rainonly=False, noshallow=False,
                     N_trn_exs=None, convcond=False, doRF=False):
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
    Returns:
        str: String id of trained NN
    """
    # Load training data
    if convcond:
        trainfile = './data/convcond_training_v3.pkl'
        testfile = './data/convcond_testing_v3.pkl'
        pp_str = 'convcond_'
    else:
        trainfile = './data/conv_training_v3.pkl'
        testfile = './data/conv_testing_v3.pkl'
        pp_str = ''
    x, y, cv, Pout, lat, lev, dlev, timestep = \
        nnload.loaddata(trainfile, minlev, rainonly=rainonly,
                        noshallow=noshallow, N_trn_exs=N_trn_exs)
    # Set up weights for training examples
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
    # Transform data according to input preprocessor requirements
    x_pp = nnload.init_pp(x_ppi, x)
    x = nnload.transform_data(x_ppi, x_pp, x)
    y_pp = nnload.init_pp(y_ppi, y)
    y = nnload.transform_data(y_ppi, y_pp, y)
    # Make preprocessor string for saving
    pp_str = pp_str + 'X-' + x_ppi['name'] + '-' + x_ppi['method'][:6] + '_'
    pp_str = pp_str + 'Y-' + y_ppi['name'] + '-' + y_ppi['method'][:6] + '_'
    # Add number of training examples to string
    pp_str = pp_str + 'Ntrnex' + str(N_trn_exs) + '_'
    if weight_decay > 0.0:
        regularize = 'L2'
    else:
        regularize = None
    if doRF:
        r_mlp, r_str = build_randomforest(500, pp_str)
        errors_stored = np.ones((1, 6))
    else:
        # Build neural network
        r_mlp, r_str = build_nn('regress', num_layers, 'Rectifier', hidneur,
                                'momentum', pp_str, batch_size=100,
                                n_stable=n_stable, n_iter=n_iter,
                                learning_momentum=0.9, learning_rate=0.01,
                                regularize=regularize,
                                weight_decay=weight_decay)
    if weight_precip:
        r_str = r_str + '_wpr'
    if weight_shallow:
        r_str = r_str + '_wsh'
    r_str = r_str + '_v3'  # reflects that we are loading v3 of training data
    # Print details about the ML algorithm we are using
    print(r_str + ' Using ' + str(x.shape[0]) + ' training examples with ' +
          str(x.shape[1]) + ' input features and ' + str(y.shape[1]) +
          ' output targets')
    # Train neural network
    r_mlp, r_errors = train_nn(r_mlp, r_str, x, y, w)
    # Save neural network
    pickle.dump([r_mlp, r_str, r_errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev,
                 dlev], open('./data/regressors/' + r_str + '.pkl', 'wb'))
    # Plot figures with validation data (and with training data)
    nnplot.plot_all_figs(r_str, datasource=testfile, noshallow=noshallow,
                         rainonly=rainonly)
    nnplot.plot_all_figs(r_str, datasource=trainfile, validation=False,
                         noshallow=noshallow, rainonly=rainonly)
    return r_str


def store_stats(i, avg_train_error, best_train_error, avg_valid_error,
                best_valid_error, avg_train_obj_error, best_train_obj_error,
                **_):
    if i == 1:
        global errors_stored
        errors_stored = []
    errors_stored.append((avg_train_error, best_train_error,
                          avg_valid_error, best_valid_error,
                          avg_train_obj_error, best_train_obj_error))


def build_nn(method, num_layers, actv_fnc, hid_neur, learning_rule, pp_str,
             batch_size=100, n_iter=None, n_stable=None,
             learning_rate=0.01, learning_momentum=0.9,
             regularize='L2', weight_decay=0.0, valid_size=0.5,
             f_stable=.001):
    """Builds a multi-layer perceptron via the scikit neural network interface
    """
    # First build layers
    actv_fnc = num_layers*[actv_fnc]
    hid_neur = num_layers*[hid_neur]
    layers = [sknn_jgd.mlp.Layer(f, units=h) for f, h in zip(actv_fnc, hid_neur)]
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
        layers.append(sknn.mlp.Layer("Softmax"))
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


def build_randomforest(N_trees, mlp_str):
    mlp = RandomForestRegressor(n_estimators=N_trees)
    mlp_str = 'RF_regress_' + str(N_trees)
    return mlp, mlp_str


def train_nn(mlp, mlp_str, x, y, w=None):
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


# ---  EVALUATING NEURAL NETS  --- #




def compare_convcond_prediction(cv_str, cvcd_str, minlev):
    cv_mlp, _, errors, x_ppi, y_ppi, x_pp, y_pp, lat, lev, _ = \
        pickle.load(open('./data/regressors/' + cv_str + '.pkl', 'rb'))
    cvcd_mlp, _, errors, x_ppi_check, y_ppi_check, x_pp, y_pp, lat, lev, _ = \
        pickle.load(open('./data/regressors/' + cvcd_str + '.pkl', 'rb'))
    # Check that preprocessers are the same
    if ((x_ppi != x_ppi_check) or (y_ppi != y_ppi_check)):
        raise ValueError('Preprocessing schemes different for conv only and ' +
                         'conv+cond!')
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
    mse_cvcd_predictboth = calc_mse(nnload.unpack(ypcvcd_scl, v), nnload.unpack(ytcvcd_scl, v), relflag=True)
    mse_cv = calc_mse(nnload.unpack(ypcv_scl, v), nnload.unpack(ytcv_scl, v), relflag=True)
    print('MSE predicting convection and condensation in one step: {:.5f}'.
          format(mse_cvcd_predictboth))
    print('MSE predicting convection only (no condensation): {:.5f}'.
          format(mse_cv))
    # Calculate MSE for both types


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

def plot_classifier_hist(pred,y,tistr):
    """Make histograms of the classification scores binned by the
        'strength' of the convection
    Inputs: pred  - predicted classification   (N_samples x 1)
            y     - true T,q tendencies        (N_samples x N_features)
            tistr - used as title and filename (str)"""
    pred = np.squeeze(pred)
    # Calculate convection strength and max it out at 100 for plotting purposes
    conpower = np.sum(np.abs(unpack(y,'T')),axis=1)
    maxbin=np.floor(.7*np.amax(conpower))
    conpower = np.clip(conpower,0,maxbin)
    # Calculate some overall statistics
    both1 = np.sum( np.logical_and(pred >0, conpower >0) ) / np.sum(conpower >0)
    both0 = np.sum( np.logical_and(pred==0, conpower==0) ) / np.sum(conpower==0)
    pct_cnvct     = np.sum(conpower >0) / len(conpower)
    pct_not_cnvct = np.sum(conpower==0) / len(conpower)
    # Limit data to times when convection is really happening
    ind = conpower>0
    pred     = pred[ind]
    conpower = conpower[ind]
    # Plot figure
    fig=plt.figure()
    bins=np.linspace(0,maxbin,100)
    plt.hist(conpower[pred==0],bins,label='wrong',alpha=0.5)
    plt.hist(conpower[pred==1],bins,label='right',alpha=0.5)
    plt.legend()
    plt.title(tistr)
    plt.xlabel('"Intensity" of Convection')
    # Write overall statistics including how well we do at classifying when convection does not occur
    plt.gca().text(.1,.5,'Classifier correctly predicts convection: %.1f%% of time (it convects %.1f%% of time)'
                   %(100.*both1,100.*pct_cnvct    ),verticalalignment='bottom', horizontalalignment='left',transform=plt.gca().transAxes)
    plt.gca().text(.1,.4,'Classifier correctly predicts no convection : %.1f%% of time (it does not convect  %.1f%% of time)'
                   %(100.*both0,100.*pct_not_cnvct),verticalalignment='bottom', horizontalalignment='left',transform=plt.gca().transAxes)
    plt.show()
    fig.savefig(fig_dir + 'convection_classifier_%s.png' %(tistr),
                bbox_inches='tight', dpi=450)

def plot_roc_curve(mlp_list, mlp_str, X, y_true):
    # For classifier
    auroc_score = []
    fig = plt.figure()
    for ind, mlp in enumerate(mlp_list):
        tp_probs = mlp.predict_proba(X)
        tp_probs = tp_probs[:,1]
        fpr, tpr, _ =      metrics.roc_curve(    y_true, tp_probs)
        auroc_score.append(metrics.roc_auc_score(y_true, tp_probs))
        plt.plot(fpr, tpr, label=mlp_str[ind])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig(fig_dir + 'classify_roc_curves.png',bbox_inches='tight',dpi=450)
    return auroc_score

def plot_classifier_metrics(mlp_list,mlp_str,X,y_true,auroc_score):
    mcc = []
    logloss = []
    tick = np.arange(len(mlp_list))
    for mlp in mlp_list:
        y_pred = mlp.predict(X)
        mcc.append(metrics.matthews_corrcoef(y_true, y_pred))
        logloss.append(metrics.log_loss(     y_true, mlp.predict_proba(X)))
    def do_plt(metric, ind, titlestr, mlp_str):
        plt.subplot(1,3,ind)
        plt.plot(metric, tick, marker = 'o')
        if ind==1:
            plt.yticks(tick, mlp_str)
        else:
            plt.setp(plt.gca().get_yticklabels(), visible=False)
        plt.title(titlestr)
        plt.tight_layout()
    fig = plt.figure(102)
    do_plt(mcc,         1, 'Matthews corr. coeff.', mlp_str)
    do_plt(auroc_score, 2, 'Area under ROC curve' , mlp_str)
    do_plt(logloss,     3, 'Cross-entropy Loss'   , mlp_str)
    plt.show()
    fig.savefig(fig_dir + 'classify_metrics.png',bbox_inches='tight',dpi=450)
