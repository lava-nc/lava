# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import warnings
from scipy.optimize import fsolve
from scipy.special import zetac
from scipy.special import erf


# Define auxiliary functions for weight conversion.
def _mean_input(num_neurons_exc, gamma, g_factor, weight, rate, bias):
    '''
    Calculate mean input to single neuron given mean excitatory weight.

    Parameters
    ----------
    num_neurons_exc : int
        Number of excitatory neurons
    gamma : float
        Ratio of inhibitory and excitatory neurons
    g_factor : float
        Factor controlling inhibition-excitation balance
    weight : float
        Mean excitatory weight
    rate : float
        Mean rate of neurons in network
    bias : float
        Bias provided to neurons

    Returns
    -------
    mean_inp : float
        Mean input received by each neuron
    '''
    mean_inp = num_neurons_exc * (1 - gamma * g_factor) * weight * rate + bias

    return mean_inp


def _std_input(num_neurons_exc, gamma, g_factor, weight, rate):
    '''
    Calculate mean input to single neuron given mean excitatory weight.

    Parameters
    ----------
    num_neurons_exc : int
        Number of excitatory neurons
    gamma : float
        Ratio of inhibitory and excitatory neurons
    g_factor : float
        Factor controlling inhibition-excitation balance
    weight : float
        Mean excitatory weight
    rate : float
        Mean rate of neurons in network

    Returns
    -------
    mean_inp : float
        Mean input received by each neuron
    '''
    return num_neurons_exc * (1 + gamma * g_factor**2) * weight ** 2 * rate


def _y_th(vth, mean, std, dv_exc, du_exc):
    '''
    Effective threshold, see Grytskyy et al. 2013.

    Parameters
    ----------
    vth : float
        Threshold of LIF neuron
    mean : float
        Mean input of neuron
    std : float
        Standard deviation of input
    dv_exc : float
        Integration constant of voltage variable
    du_exc : float
        Integration constant of current variable

    Returns
    -------
    yth : float
        Effective threshold of neuron in network
    '''
    y_th = (vth - mean) / std
    y_th += np.sqrt(2) * np.abs(zetac(0.5)) * np.sqrt(dv_exc / du_exc) / 2

    return y_th


def _y_r(mean, std, dv_exc, du_exc):
    '''
    Effective reset, see Grytskyy et al. 2013.

    Parameters
    ----------
    vth : float
        Threshold of LIF neuron
    mean : float
        Mean input of neuron
    std : float
        Standard deviation of input
    dv_exc : float
        Integration constant of voltage variable
    du_exc : float
        Integration constant of current variable

    Returns
    -------
    yr : float
        Effective reset of neuron in network
    '''
    y_r = (- 1 * mean) / std
    y_r += np.sqrt(2) * np.abs(zetac(0.5)) * np.sqrt(dv_exc / du_exc) / 2

    return y_r


def f(y):
    '''
    Derivative of transfer function of LIF neuron at given argument.
    '''
    return np.exp(y ** 2) * (1 + erf(y))


def _alpha(vth, mean, std, dv_exc, du_exc):
    '''
    Auxiliary variable describing contribution of weights for weight
    mapping given network state, see Grytskyy et al. 2013.

    Parameters
    ----------
    vth : float
        Threshold of LIF neuron
    mean : float
        Mean input of neuron
    std : float
        Standard deviation of input
    dv_exc : float
        Integration constant of voltage variable
    du_exc : float
        Integration constant of current variable

    Returns
    -------
    val : float
        Contribution of weight
    '''
    val = np.sqrt(np.pi) * (mean * dv_exc * 0.01) ** 2
    val *= 1 / std
    val *= (f(_y_th(vth, mean, std, dv_exc, du_exc))
            - f(_y_r(mean, std, dv_exc, du_exc)))

    return val


def _beta(vth, mean, std, dv_exc, du_exc):
    '''
    Auxiliary variable describing contribution of square of weights for
    weight mapping given network state, see Grytskyy et al. 2013.

    Parameters
    ----------
    vth : float
        Threshold of LIF neuron
    mean : float
        Mean input of neuron
    std : float
        Standard deviation of input
    dv_exc : float
        Integration constant of voltage variable
    du_exc : float
        Integration constant of current variable

    Returns
    -------
    val : float
        Contribution of square of weights
    '''
    val = np.sqrt(np.pi) * (mean * dv_exc * 0.01) ** 2
    val *= 1 / (2 * std ** 2)
    val *= (f(_y_th(vth, mean, std, dv_exc, du_exc)) * (vth - mean) / std
            - f(_y_r(mean, std, dv_exc, du_exc)) * (-1 * mean) / std)

    return val


def convert_rate_to_lif_params(
        shape_exc, dr_exc, bias_exc, shape_inh, dr_inh, bias_inh, g_factor,
        q_factor, weights, **kwargs):
    '''Convert rate parameters to LIF parameters.
    The mapping is based on A unified view on weakly correlated recurrent
    network, Grytskyy et al. 2013.

    Parameters
    ----------
    shape_exc : int
        Number of excitatory neurons in rate network
    dr_exc : float
        Integration constant for excitatory neurons in rate network
    bias_exc : float
        Bias for excitatory neurons in rate network
    shape_inh : int
        Number of inhibitory neurons in rate network
    dr_inh : float
        Integration constant for inhibitory neurons in rate network
    bias_inh : float
        Bias for inhibitory neurons in rate network
    g_factor : float
        Factor controlling inhibition-excitation balance
    q_factor : float
        Factor controlling response properties of rate network
    weights : np.ndarray
        Recurrent weights of rate network

    Returns
    -------
    lif_network_dict : dict
        Parameter dictionary for LIF network
    '''
    # Copy weight parameters.
    weights_local = weights.copy()

    num_neurons_exc = shape_exc
    num_neurons_inh = shape_inh

    # Ratio of excitatory to inhibitory neurons.
    gamma = float(num_neurons_exc) / float(num_neurons_inh)

    # Assert that network is balanced.
    if gamma * g_factor <= 1:
        raise AssertionError("Network not balanced, increase g_factor")

    # Set timescales of neurons.
    dv_exc = 1 * dr_exc  # Dynamics of membrane potential as fast as rate.
    du_exc = 7 * dr_exc  # Dynamics of current 7 times as fast as rate.

    dv_inh = 1 * dr_inh  # Dynamics of membrane potential as fast as rate.
    du_inh = 7 * dr_inh  # Dynamics of current 7 times as fast as rate.

    # Set threshold to default value.
    vth_exc = 1
    vth_inh = 1

    # Set biases.
    # First  calculate relative biases for rate model.
    if bias_exc >= bias_inh:
        rel_exc_inh_bias = bias_exc / bias_inh
        rel_inh_exc_bias = 1
    else:
        rel_inh_exc_bias = bias_inh / bias_exc
        rel_exc_inh_bias = 1

    # We then determine the the bias for the LIF network.
    # We have to be careful not the reduce the bias since a too small bias
    # results in inactivity.
    bias_exc = 5 * vth_exc * dv_exc * rel_exc_inh_bias
    bias_inh = 5 * vth_inh * dv_inh * rel_inh_exc_bias

    # Get the mean excitatory weight.
    exc_weights = weights_local[:, :num_neurons_exc]
    mean_exc_weight = np.mean(exc_weights)

    # Perform weight conversion.

    # First determine approximately stationary firing rate in inhibition
    # dominated regime.
    # See Dynamic of Sparsely Connected Networks of Excitatory and
    # Inhibitory Spiking Neurons, Brunel, 2000.
    # We simplify the calculation by working with average acitivites.
    bias = (bias_exc / dv_exc + bias_inh / dv_inh) / 2
    rate = (bias - vth_exc) / (gamma * g_factor - 1)

    # Function describing mapping of rate to LIF weights problem about
    # finding a zero.
    def func(weight):
        '''
        Adapted from Grytskyy et al..
        '''
        mean_inp = _mean_input(num_neurons_exc, gamma,
                               g_factor, weight, rate, bias)
        std_inp = _std_input(num_neurons_exc, gamma,
                             g_factor, weight, rate)
        alpha = _alpha(vth_exc, mean_inp, std_inp, dv_exc, du_inh)
        beta = _beta(vth_exc, mean_inp, std_inp, dv_exc, du_inh)

        return mean_exc_weight - alpha * weight - beta * weight**2

    # Solve for weights of LIF network retaining correlation structure of
    # rate network.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '', RuntimeWarning)
        try:
            mean_exc_weight_new = fsolve(func, mean_exc_weight)[0]
            # Determine weight scaling factor
            weight_scale = mean_exc_weight_new / mean_exc_weight
        except Warning:
            # Theory breaks done, most likely due to strong correlations
            # induced by strong weights. Choose 1 as scaling factor.
            weight_scale = 1

    # Scale weights.
    if weight_scale > 0:
        weights_local *= weight_scale
    else:
        print('Weigh scaling factor not positive: No weight scaling possible')

    # Scale weights with integration time step.
    weights_local[:, :num_neurons_exc] *= du_exc
    weights_local[:, num_neurons_exc:] *= du_inh

    # Single neuron paramters.
    # Bias_mant is set to make the neuron spike.
    lif_params_exc = {
        "shape_exc": num_neurons_exc,
        "vth_exc": vth_exc,
        "du_exc": du_exc,
        "dv_exc": dv_exc,
        "bias_mant_exc": bias_exc}

    lif_params_inh = {
        "shape_inh": num_neurons_inh,
        "vth_inh": vth_inh,
        "du_inh": du_inh,
        "dv_inh": dv_inh,
        "bias_mant_inh": bias_inh}

    # Parameters Paramters for E/I network/
    network_params_lif = {}

    network_params_lif.update(lif_params_exc)
    network_params_lif.update(lif_params_inh)
    network_params_lif['g_factor'] = g_factor
    network_params_lif['q_factor'] = q_factor
    network_params_lif['weights'] = weights_local

    return network_params_lif
