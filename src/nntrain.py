import numpy as np
import sknn.mlp
import time
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import src.nnload as nnload
import pickle

# ---  BUILDING NEURAL NETS  --- #

def train_nn_wrapper(hidneur, n_iter=None, n_stable=None,
                     minlev=0.0,rainonly=False,data_dir='./data/'):
    #Import packages
    # Load data
    x, y, cv, Pout, lat, lev, dlev, timestep = nnload.loaddata(data_dir + 'nntest.nc', minlev,
                                                           rainonly=rainonly) #,all_lats=False,indlat=8)
    # Preprocess data
    scaler_x, scaler_y, x1, x2, x3, y1, y2, y3, cv1, cv2, cv3 = nnload.pp(x, y, cv, 30000)
    # Build, train, and save model
    r_mlp, r_str = build_nn('regress',['Rectifier'],[hidneur],
                                    'momentum',batch_size=100,n_stable=n_stable,n_iter=n_iter)
                                #, regularize = 'L2', weight_decay = 0.0001)
    print(r_str)
    r_mlp, r_errors = train_nn(r_mlp,r_str,x1,y1,x2,y2)
    pickle.dump([r_mlp, r_str, r_errors], 
                open(data_dir + 'regressors/' + r_str + '.pkl', 'wb'))

error=[]
def store_stats( avg_train_error, best_train_error, avg_valid_error, best_valid_error,**_):
    error.append((avg_train_error, best_train_error, avg_valid_error, best_valid_error))

def build_nn(method,actv_fnc,hid_neur,learning_rule,batch_size=100,n_iter=None, n_stable=None,
             learning_rate=0.01,learning_momentum=0.9,
             regularize = None, weight_decay = None, valid_size = 0.25):
    """Builds a multi-layer perceptron via the scikit neural network interface"""
    # First build layers
    layers = [sknn.mlp.Layer(f,units=h) for f,h in zip(actv_fnc,hid_neur)]
    # Append a linear output layer if regressing and a softmax layer if classifying
    if method == 'regress':
        layers.append(sknn.mlp.Layer("Linear"))
        mlp = sknn.mlp.Regressor(layers,n_iter=n_iter,batch_size=batch_size,learning_rule=learning_rule,
                                 learning_rate=learning_rate,learning_momentum=learning_momentum,
                                 regularize = regularize, weight_decay = weight_decay, n_stable=n_stable,
                                 valid_size=valid_size, 
                                 callback={'on_epoch_finish': store_stats})
    if method == 'classify':
        layers.append(sknn.mlp.Layer("Softmax"))
        mlp = sknn.mlp.Classifier(layers,n_iter=n_iter,batch_size=batch_size,learning_rule=learning_rule,
                                 learning_rate=learning_rate,learning_momentum=learning_momentum,
                                 regularize = regularize, weight_decay = weight_decay, n_stable=n_stable,
                                 valid_size=valid_size, callback={'on_epoch_finish': store_stats})
    #Write nn string
    layerstr='_'.join([str(h)+f[0] for h,f in zip(hid_neur,actv_fnc)])
    mlp_str = method[0] + "_" +  layerstr + "_" + learning_rule[0:3] + "{}".format(str(learning_momentum) if learning_rule=='momentum'
                                                                      else str(learning_rate))
    if regularize is not None:
        mlp_str = mlp_str + 'reg' + str(weight_decay)
    return mlp,mlp_str

def _build_classifiers(c_mlp,c_str,n_iter=10000):
    # OUTDATED - NEEDS TO BE UPDATED OR REMOVED
    """Returns a list of classifier MLP objects and a str with their abbreivated names"""
    batch_size=100
    c_mlp.append(build_nn('classify',['Rectifier','Rectifier']            ,[500,500]    ,n_iter,batch_size,'momentum'))
    c_mlp.append(build_nn('classify',['Rectifier','Rectifier','Rectifier'],[200,200,200],n_iter,batch_size,'momentum'))
    c_mlp.append(build_nn('classify',['Tanh','Tanh','Tanh']               ,[200,200,200],n_iter,batch_size,'momentum'))
    c_mlp.append(build_nn('classify',['Tanh','Tanh']                      ,[500,500]    ,n_iter,batch_size,'momentum'))
    c_mlp.append(build_nn('classify',['Tanh']                             ,[500]        ,n_iter,batch_size,'momentum'))
    c_mlp.append(build_nn('classify',['Tanh','Tanh']                      ,[500,500]    ,n_iter,batch_size,'momentum',learning_momentum=0.7))
    c_mlp.append(build_nn('classify',['Tanh','Tanh']                      ,[500,500]    ,n_iter,batch_size,'sgd'))
    c_mlp,c_str = map(list, zip(*c_mlp))
    return c_mlp, c_str

def build_randomforest(method,mlp,mlp_str):
    methods = {'classify':RandomForestClassifier,'regress':RandomForestRegressor}
    estimators = {'classify':[200,50,20],'regress':[200,50,20]}
    for estimator in estimators[method]:
        mlp.append(methods[method](n_estimators=estimator))
        mlp_str.append(method[0] + '_RF_' + str(estimator))
    return mlp, mlp_str

def train_nn(mlp,mlp_str,x1,cv1,x2,cv2,y2=None):
    """Train each item in a list of multi-layer perceptrons and then score on test data"""
    # Expects a mulit-layer perceptron object
    # Initialize
    n_iter = mlp.n_iter
    start = time.time()
    # Train the model using training data
    mlp.fit(x1,cv1)
    train_score = mlp.score(x1, np.squeeze(cv1))
    test_score  = mlp.score(x2, np.squeeze(cv2))
    end = time.time()
    print("Training Score: {:.4f}, Test Score: {:.4f} for Model {:s} ({:.1f} seconds)".format(
                                              train_score, test_score, mlp_str, end-start))
    errors = np.asarray(error) # This is an N_iter x 4 array...see score_stats
    # Return the fitted models and the scores
    return mlp, errors


# ---  EVALUATING NEURAL NETS  --- #

def plot_model_error_over_time(errors, mlp_str, fig_dir, txt):
    x = np.arange(errors.shape[0])
    ytix = [.5e-3,1e-3,2e-3,5e-3,10e-3,20e-3]
    # Plot error rate vs. iteration number
    fig=plt.figure()
    # Plot training errors
    ax1=plt.subplot(1, 2 ,1)
    plt.semilogy(x, np.squeeze(errors[:,0,:]), alpha=0.5)
    plt.legend(mlp_str)
    plt.gca().set_color_cycle(None) #reset color cycle
    plt.semilogy(x, np.squeeze(errors[:,1,:]), alpha=0.5)
    plt.yticks(ytix,ytix)
    plt.ylim((np.nanmin(errors[:,0,:]), np.nanmax(errors[:,0,:])))
    plt.title('Training Error')
    # Plot test errors
    plt.subplot(1, 2 ,2)
    plt.semilogy(x, np.squeeze(errors[:,2,:]), alpha=0.5)
    plt.legend(mlp_str)
    plt.gca().set_color_cycle(None) #reset color cycle
    plt.semilogy(x, np.squeeze(errors[:,3,:]), alpha=0.5)
    plt.yticks(ytix,ytix)
    plt.ylim((np.nanmin(errors[:,2,:]), np.nanmax(errors[:,2,:])))
    plt.title('Testing Error')
    plt.show()
    fig.savefig(fig_dir + txt + '_score_history.png',bbox_inches='tight',dpi=450)
            
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
    plt.plot(mse, tick, marker='o',)
    plt.title('Mean Squared Error')
    # Plot R2 
    plt.subplot(1,2,2)
    plt.plot(r2_u, tick, marker='o', label='uniform')
    plt.plot(r2_w, tick, marker='o', label='weighted')
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.legend(loc="upper left")
    plt.title('R^2 score')
    fig.savefig(fig_dir + txt + '_scores.png',bbox_inches='tight',dpi=450)

def classify(classifier,x,y):
    """Applies a trained classifier to input x and y and 
    outputs data when classifier expects convection is occurring."""
    # Apply classifier 
    out = classifier.predict(x)
    # Get samples of when classifier predicts that it is convecting
    ind = np.squeeze(np.not_equal(out,0))
    # Store data only when it is convection is predicted
    x = x[ind,:]
    y = y[ind,:]
    return x,y
             
def plot_classifier_hist(pred,y,tistr):
    """Make histograms of the classification scores binned by the 'strength' of the convection
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
    fig.savefig(fig_dir + 'convection_classifier_%s.png' %(tistr),bbox_inches='tight',dpi=450)
    
def plot_roc_curve(mlp_list,mlp_str,X,y_true):
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



