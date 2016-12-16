import numpy as np

# ----  HELPER SCRIPTS  ---- #


def calc_enthalpy(T, q, dlev):
    # y is output data set in rate (1/day)
    # k is the implied uniform heating rate over the whole column to correct
    # the imbalance
    cp = 1005.  # J/kg/K
    L = 2.5e6  # J/kg
    k = (T + (L/cp) * q/1000.)
    k = vertical_integral(k, dlev)
    k = k / 1e5
    return k


def vertical_integral(data, dlev):
    g = 9.8  # m/s2
    data = -1/g * np.sum(data * dlev[:, None].T, axis=1)*1e5
    return data


def calc_precip(q, dlev):
    q = q / 1000  # kg/kg/day
    return vertical_integral(q, dlev)  # mm/day


def calc_theta(T, sigma):
    kappa = 287/1005
    theta = T * np.power(1. / sigma, kappa)
    return theta


def calc_theta_e(T, theta, q):
    L = 2.5e6
    Cp = 1005
    theta_e = theta * np.exp(L * q / Cp / T)
    return theta_e
