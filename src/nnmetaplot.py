import numpy as np
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt

# ----  META-PLOTTING SCRIPTS  ---- #


def load_r_mlps():
    neur_strL = ['5R', '10R', '5R_5R', '50R', '100R', '10R_10R', '200R',
                 '50R_50R', '100R_100R', '200R_200R']
    neur_str = ['5', '10', '5-5', '50', '100', '10-10', '200',
                '50-50', '100-100', '200-200']
    neur_val = np.array([5, 10, 25, 50, 100, 125, 200, 2500, 1e4, 4e4])
    trn_ex = np.array([1000, 5000, 10000, 100000, 400000])
    regs = np.array([1e-7, 1e-6, 1e-5])
    tr = np.nan * np.zeros((len(neur_str), len(trn_ex), len(regs)))
    cv = np.nan * np.zeros((len(neur_str), len(trn_ex), len(regs)))
    ptf = 'X-StandardScaler-qTindi_Y-SimpleY-qTindi_Ntrnex'
    for i, hid in enumerate(neur_strL):
        for j, nex in enumerate(trn_ex):
            for k, reg in enumerate(regs):
                r_str = ptf + str(nex) + '_r_' + hid + \
                    '_mom0.9reg' + str(reg) + '_Niter10000_v3'
                try:
                    _, _, err, _, _, _, _, _, _, _ = \
                        pickle.load(open('./data/regressors/' + r_str + '.pkl',
                                         'rb'))
                    tr[i, j, k] = err[-1, 4]
                    cv[i, j, k] = err[-1, 2]
                except FileNotFoundError:
                    None
                    # print(r_str + ' was not found.')
    return tr, cv, neur_str, neur_val, trn_ex, regs


def meta_plot_regs():
    tr, cv, neur_str, neur_val, trn_ex, regs = load_r_mlps()
    fig, ax = plt.subplots(len(neur_str), len(trn_ex))
    for i, neur_s in enumerate(neur_str):
        for j, trn_e in enumerate(trn_ex):
            masks = np.isfinite(tr[i, j, :])
            ax[i, j].semilogx(regs[masks], tr[i, j, masks], marker='o')
            ax[i, j].semilogx(regs[masks], cv[i, j, masks], marker='o',
                              color='green')
            ax[i, j].text(.2, .7, neur_s + ' ' + str(trn_e),
                          transform=ax[i, j].transAxes)
            ax[i, j].set_xlim(regs[0], regs[-1])
            ax[i, j].get_xaxis().set_ticks([])
            ax[i, j].get_yaxis().set_ticks([])
    fig.savefig('./figs/NN_eval_vs_reg.eps', bbox_inches='tight')


def meta_compare_error_rate_v2():
    tr, cv, neur_str, neur_val, trn_ex, regs = load_r_mlps()
    tr = np.squeeze(tr[:, :, regs == 1e-6])
    cv = np.squeeze(cv[:, :, regs == 1e-6])
    # Plot as a function of number of hidden neurons
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    colormat = plt.cm.plasma(np.linspace(0, 1, tr.shape[1]))
    for i in range(tr.shape[1]):
        # Indicate where missing values are for plotting
        tr_mask = np.isfinite(tr[:, i])
        cv_mask = np.isfinite(cv[:, i])
        print(neur_val[tr_mask])
        ax1.semilogx(neur_val[tr_mask], tr[tr_mask, i], marker='o',
                     color=colormat[i, :], label='m={:d}'.format(trn_ex[i]))
        ax2.semilogx(neur_val[cv_mask], cv[cv_mask, i], marker='o',
                     color=colormat[i, :], label='m={:d}'.format(trn_ex[i]))
    for ax in [ax1, ax2]:
        ax.set_xticks(neur_val)
        ax.set_xticklabels(neur_str, rotation='vertical')
    ax1.legend(fontsize=16)
    ax1.set_xlim(0.9*min(neur_val), 1.1*max(neur_val))
    ax1.set_title('(a) Training Error', fontsize=18)
    ax2.set_title('(b) Cross-Validation Error', fontsize=18)
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=16)
    fig.text(0.5, -0.1, 'Number of Hidden Neurons', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Mean Squared Error', va='center', rotation='vertical',
             fontsize=18)
    fig.savefig('./figs/NN_eval_vs_h.eps', bbox_inches='tight')
    # Plot as a function of number of training examples
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    colormat = plt.cm.plasma(np.linspace(0, 1, tr.shape[0]))
    for i in range(tr.shape[0]):
        tr_mask = np.isfinite(tr[i, :])
        cv_mask = np.isfinite(cv[i, :])
        ax1.semilogx(trn_ex[tr_mask]*0.5, tr[i, tr_mask], marker='o',
                     color=colormat[i, :], label=neur_str[i])
        ax2.semilogx(trn_ex[cv_mask]*0.5, cv[i, cv_mask], marker='o',
                     color=colormat[i, :], label=neur_str[i])
    ax1.legend(fontsize=16, ncol=2)
    ax1.set_xlim(0.9*min(trn_ex*0.5), 1.1*max(trn_ex*0.5))
    ax1.set_ylim(0, 0.4)
    ax1.set_title('(a) Training Error', fontsize=18)
    ax2.set_title('(b) Cross-Validation Error', fontsize=18)
    fig.text(0.5, 0.04, 'Number of Training Examples', ha='center',
             fontsize=18)
    fig.text(0.04, 0.5, 'Mean Squared Error', va='center', rotation='vertical',
             fontsize=18)
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=16)
    plt.show()
    fig.savefig('./figs/NN_eval_vs_m.eps', bbox_inches='tight')
    # plt.figure()
    # for i, mark in enumerate(markers):
    #     plt.plot(mse[:, i], ls='-', marker=mark, color='blue',
    #              label='mse, N={:d}'.format(trn_ex[i]))
    # plt.show()


def meta_compare_error_rate():
    hid_neur = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150]
    e = dict()
    e['train_L1_R0'] = []
    e['test_L1_R0'] = []
    e['train_L2_R0'] = []
    e['test_L2_R0'] = []
    e['train_L1_R1e-5'] = []
    e['test_L1_R1e-5'] = []
    e['train_L2_R1e-5'] = []
    e['test_L2_R1e-5'] = []
    ptf = './data/regressors/X-StandardScaler-qTindi_Y-SimpleY-qTindi_r_'
    for h in hid_neur:
        hs = str(h)
        _, _, e_load, _, _, _, _, _, _, _ = \
            pickle.load(open(ptf + hs + 'R_mom0.9.pkl', 'rb'))
        e['train_L1_R0'].append(np.amin(e_load[-1, 1]))
        e['test_L1_R0'].append(np.amin(e_load[-1, 3]))
        _, _, e_load, _, _, _, _, _, _, _ = \
            pickle.load(open(ptf + hs + 'R_' + hs + 'R_mom0.9.pkl', 'rb'))
        e['train_L2_R0'].append(np.amin(e_load[-1, 1]))
        e['test_L2_R0'].append(np.amin(e_load[-1, 3]))
        _, _, e_load, _, _, _, _, _, _, _ = \
            pickle.load(open(ptf + hs + 'R_mom0.9reg1e-05.pkl', 'rb'))
        e['train_L1_R1e-5'].append(np.amin(e_load[-1, 1]))
        e['test_L1_R1e-5'].append(np.amin(e_load[-1, 3]))
        _, _, e_load, _, _, _, _, _, _, _ = \
            pickle.load(open(ptf + hs + 'R_' + hs + 'R_mom0.9reg1e-05.pkl',
                        'rb'))
        e['train_L2_R1e-5'].append(np.amin(e_load[-1, 1]))
        e['test_L2_R1e-5'].append(np.amin(e_load[-1, 3]))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(hid_neur, e['train_L1_R0'], color='b', ls='-', marker='o',
             label='Train 1-layer')
    ax1.plot(hid_neur, e['test_L1_R0'], color='r', ls='-', marker='o',
             label='Test 1-layer')
    ax1.plot(hid_neur, e['train_L2_R0'], color='b', ls='--', marker='s',
             label='Train 2-layer')
    ax1.plot(hid_neur, e['test_L2_R0'], color='r', ls='--', marker='s',
             label='Test 2-layer')
    ax1.set_title('Error rate (no regularization)')
    ax2.plot(hid_neur, e['train_L1_R1e-5'], color='b', ls='-', marker='o',
             label='Train 1-layer')
    ax2.plot(hid_neur, e['test_L1_R1e-5'], color='r', ls='-', marker='o',
             label='Test 1-layer')
    ax2.plot(hid_neur, e['train_L2_R1e-5'], color='b', ls='--', marker='s',
             label='Train 2-layer')
    ax2.plot(hid_neur, e['test_L2_R1e-5'], color='r', ls='-', marker='s',
             label='Test 2-layer')
    ax2.legend(loc='upper right')
    ax2.set_title('Error Rate (with regularization)')
    plt.show()
    fig.savefig('./figs/Compare_error_rate_vs_hid_neur.png',
                bbox_inches='tight', dpi=450)


def meta_plot_model_error_vs_training_examples():
    n_samp = np.array([100, 200, 500, 1000, 2000, 3500, 5000, 7500, 10000,
                       12500, 15000])
    err_trn = []
    err_tst = []
    # Load data for each sample
    for n in n_samp:
        r_str = 'X-StandardScaler-qTindi_Y-SimpleY-qTindi_Ntrnex' + str(n) + \
                '_r_60R_60R_mom0.9_Niter10000'
        _, _, errors, _, _, _, _, _, _, _ = \
            pickle.load(open('./data/regressors/' + r_str + '.pkl', 'rb'))
        err_trn.append(np.amin(errors[:, 0]))
        err_tst.append(np.amin(errors[:, 2]))
    fig = plt.figure()
    n_samp = n_samp / 2.  # we are doing a 50-50 split for train & cross-val
    plt.plot(n_samp, err_trn, label='Train')
    plt.plot(n_samp, err_tst, label='Test')
    plt.legend()
    r_str_save = 'X-StandardScaler-qTindi_Y-SimpleY-qTindi' + \
                 '_r_60R_60R_mom0.9_Niter10000'
    plt.title('Error Rate for ' + r_str_save)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error Rate')
    fig.savefig('./figs/' + r_str_save + 'error_history.png',
                bbox_inches='tight', dpi=450)


def plot_regressors_scores(r_list, r_str, x_test, y_true, fig_dir, txt):
    """Given a list of fitted regressor objects, compare their skill on a
    variety of tests"""
    mse = []
    r2_u = []
    r2_w = []
    exp_var_u = []
    exp_var_w = []
    for reg in r_list:
        y_pred = reg.predict(x_test)
        mse.append(metrics.mean_squared_error(y_true, y_pred,
                                              multioutput='uniform_average'))
        r2_u.append(metrics.r2_score(y_true, y_pred,
                                     multioutput='uniform_average'))
        r2_w.append(metrics.r2_score(y_true, y_pred,
                                     multioutput='variance_weighted'))
        expl = metrics.explained_variance_score
        exp_var_u.append(expl(y_true, y_pred, multioutput='uniform_average'))
        exp_var_w.append(expl(y_true, y_pred, multioutput='variance_weighted'))
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    tick = range(len(mse))
    # Plot mean squared error
    plt.yticks(tick, r_str)
    plt.semilogx(mse, tick, marker='o',)
    plt.title('Mean Squared Error')
    # Plot R2
    plt.subplot(1, 2, 2)
    plt.plot(r2_u, tick, marker='o', label='uniform')
    plt.plot(r2_w, tick, marker='o', label='weighted')
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.legend(loc="upper left")
    plt.title('R^2 score')
    plt.xlim((-1, 1))
    fig.savefig(fig_dir + txt + '_scores.png', bbox_inches='tight', dpi=450)
    plt.close()
