import numpy as np
import warnings
from scipy.optimize import fsolve
from scipy.special import zetac
from scipy.special import erf


# Method to convert parameters from rate to LIF
def convert_rate_to_lif_params(**kwargs):
    '''Convert rate parameters to LIF parameters.
    The mapping is based on A unified view on weakly correlated recurrent
    network, Grytskyy et al. 2013

    Parameters
    ----------
    kwargs : dict
        Parameter dictionary for rate network

    Returns
    -------
    lif_network_dict : dict
        Parameter dictionary for LIF network
    '''
    # Fetch rate parameters
    shape_exc = kwargs['shape_exc']
    dr_exc = kwargs['dr_exc']
    bias_exc = kwargs['bias_exc']

    shape_inh = kwargs['shape_inh']
    dr_inh = kwargs['dr_inh']
    bias_inh = kwargs['bias_inh']

    g_factor = kwargs['g_factor']
    q_factor = kwargs['q_factor']

    weights = kwargs['weights'].copy()

    num_neurons_exc = shape_exc
    num_neurons_inh = shape_inh

    # ratio of excitatory to inhibitory neurons
    gamma = float(num_neurons_exc) / float(num_neurons_inh)

    # assert that network is balanced
    assert gamma * g_factor > 1, "Network not balanced, increase g_factor"

    # Set timescales of neurons
    dv_exc = 1 * dr_exc  # dynamics of membrane potential as fast as rate
    du_exc = 7 * dr_exc  # dynamics of current 7 times as fast as rate

    dv_inh = 1 * dr_inh  # dynamics of membrane potential as fast as rate
    du_inh = 7 * dr_inh  # dynamics of current 7 times as fast as rate

    # set threshold to default value
    vth_exc = 1
    vth_inh = 1

    # Set biases
    # First  calculate relative biases for rate model
    if bias_exc >= bias_inh:
        rel_exc_inh_bias = bias_exc / bias_inh
        rel_inh_exc_bias = 1
    else:
        rel_inh_exc_bias = bias_inh / bias_exc
        rel_exc_inh_bias = 1

    # We then determine the the bias for the LIF network.
    # We have to be careful not the reduce the bias since a too small bias
    # results in inactivity
    bias_exc = 5 * vth_exc * dv_exc * rel_exc_inh_bias
    bias_inh = 5 * vth_inh * dv_inh * rel_inh_exc_bias

    # Get the mean excitatory weight
    exc_weights = weights[:, :num_neurons_exc]
    mean_exc_weight = np.mean(exc_weights)

    # Perform weight conversion

    # First determine approximately stationary firing rate in inhibition
    # dominated regime.
    # See Dynamic of Sparsely Connected Networks of Excitatory and
    # Inhibitory Spiking Neurons, Brunel, 2000.
    # We simplify the calculation by working with average acitivites
    bias = (bias_exc / dv_exc + bias_inh / dv_inh) / 2
    rate = (bias - vth_exc) / (gamma * g_factor - 1)

    # Define auxiliary functions for weight conversion
    def _mean_input(weight):
        '''
        Calculate mean input to single neuron given mean exciatory weight
        '''
        return num_neurons_exc * (1 - gamma * g_factor) * weight * rate + bias

    def _std_input(weight):
        '''
        Calculate mean input to single neuron given mean exciatory weight
        '''
        return num_neurons_exc * (1 + gamma * g_factor**2) * weight ** 2 * rate

    def _y_th(vth, mean, std):
        '''
        Effective threshold, see Grytskyy et al.

        Parameters
        ----------
        vth : float
            Threshold of LIF neuron
        mean : float
            Mean input of neuron
        std : float
            Standard deviation of input

        Returns
        -------
        yth : float
            Effective threshold of neuron in network
        '''
        y_th = (vth - mean) / std
        y_th += np.sqrt(2) * np.abs(zetac(0.5)) * np.sqrt(dv_exc / du_exc) / 2

        return y_th

    def _y_r(mean, std):
        '''
        Effective reset, Grytskyy et al.

        Parameters
        ----------
        vth : float
            Threshold of LIF neuron
        mean : float
            Mean input of neuron
        std : float
            Standard deviation of input

        Returns
        -------
        yr : float
            Effective reset of neuron in network
        '''
        y_r = (- 1 * mean) / std
        y_r += np.sqrt(2) * np.abs(zetac(0.5)) * np.sqrt(dv_exc / du_exc) / 2

        return y_r

    # Derivative of transfer function of LIF neuron
    f = lambda y: np.exp(y**2) * (1 + erf(y))

    def _alpha(vth, mean, std):
        '''
        Auxiliary variable describing contribution of weights for weight
        mapping given network state.
        See Grytskyy et al.

        Parameters
        ----------
        vth : float
            Threshold of LIF neuron
        mean : float
            Mean input of neuron
        std : float
            Standard deviation of input

        Returns
        -------
        val : float
            Contribution of weight
        '''
        val = np.sqrt(np.pi) * (mean * dv_exc * 0.01) ** 2
        val *= 1 / std
        val *= f(_y_th(vth, mean, std)) - f(_y_r(mean, std))

        return val

    def _beta(vth, mean, std):
        '''
        Auxiliary variable describing contribution of square of weights for
        weight mapping given network state.
        See Grytskyy et al.

        Parameters
        ----------
        vth : float
            Threshold of LIF neuron
        mean : float
            Mean input of neuron
        std : float
            Standard deviation of input

        Returns
        -------
        val : float
            Contribution of square of weights
        '''
        val = np.sqrt(np.pi) * (mean * dv_exc * 0.01) ** 2
        val *= 1/(2 * std ** 2)
        val *= (f(_y_th(vth, mean, std)) * (vth - mean) / std
                - f(_y_r(mean, std)) * (-1 * mean) / std)

        return val

    # Function describing mapping of rate to LIF weights problem about
    # finding a zero
    def func(weight):
        '''
        Adapted from Grytskyy et al..
        '''
        alpha = _alpha(vth_exc, _mean_input(weight), _std_input(weight))
        beta = _beta(vth_exc, _mean_input(weight), _std_input(weight))

        return mean_exc_weight - alpha * weight - beta * weight**2

    # Solve for weights of LIF network retaining correlation structure of
    # rate network
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

    # Scale weights
    if weight_scale > 0:
        weights *= weight_scale
    else:
        print('Weigh scaling factor not positive: No weight scaling possible')

    # Scale weights with integration time step
    weights[:, :num_neurons_exc] *= du_exc
    weights[:, num_neurons_exc:] *= du_inh

    # Single neuron paramters
    # Bias_mant is set to make the neuron spike
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

    # Parameters Paramters for E/I network
    network_params_lif = {}

    network_params_lif.update(lif_params_exc)
    network_params_lif.update(lif_params_inh)
    network_params_lif['g_factor'] = g_factor
    network_params_lif['q_factor'] = q_factor
    network_params_lif['weights'] = weights

    return network_params_lif
