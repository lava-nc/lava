# copyright (c) 2021-22 intel corporation
# spdx-license-identifier: bsd-3-clause
# see: https://spdx.org/licenses/
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.process.process import AbstractProcess
from lava.magma.compiler.compiler_graphs import (
    find_processes,
    ProcGroupDiGraphs)
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import AbstractSrcPort
from lava.proc.monitor.process import Monitor
from lava.proc import io
from lava.magma.core.process.ports.ports import OutPort

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import inspect


class Float2FixedConverter:
    """
    Tool to perform automated floating- to fixed-point conversion of
    Lava Processes. To be able to perform the conversion, a floating- and
    fixed-point ProcessModel of the used Processes need to be provided.
    The converter runs the passed processes and maps the values optimally to a
    predefined fixed-point domain respecting the dynamical range of the
    variables as well as the representation constraints.

    A simple usage example - assuming procs already have been instantiated -
    may look like:

    >>> converter = Float2FixedConverter()
    >>> converter.set_run_cfg(
            fixed_pt_run_cfg=Loihi1SimCfg(select_tag='fixed_pt'),
            floating_pt_run_cfg=Loihi1SimCfg())
    >>> converter.convert(procs, num_steps=200)
    """

    def __init__(self):
        self.floating_pt_run_cfg = None
        self.fixed_pt_run_cfg = None
        self.procs = None
        self.hierarchical_procs = None
        self.run_proc = None
        self.var_ports = None
        self.fixed_pt_proc_models = None
        self.num_steps = None
        self.quantiles = None
        self.scale_domains = set()
        self.ignored_procs = [Monitor, io.sink.RingBuffer,
                              io.source.RingBuffer]

    def set_run_cfg(self, floating_pt_run_cfg, fixed_pt_run_cfg) -> None:
        """Set run config for floating- and fixed-point ProcessModels.

        Parameters
        ----------

        floating_pt_run_cfg : RunConfig
            RunConfig for floating-point ProcessModel
        fixed_pt_run_cfg : RunConfig
            RunConfig for fixed-point ProcessModel
        """

        if not issubclass(type(floating_pt_run_cfg), RunConfig):
            raise TypeError("'floating_pt_run_cfg' must be RunConfig")

        if not issubclass(type(fixed_pt_run_cfg), RunConfig):
            raise TypeError("'fixed_pt_run_cfg' must be RunConfig")

        self.floating_pt_run_cfg = floating_pt_run_cfg
        self.fixed_pt_run_cfg = fixed_pt_run_cfg

    def convert(self, proc, num_steps=100, quantiles=(0, 1),
                find_connected_procs=True):
        """Convert Processes passed from floating-point to fixed-point model.
        The floating- and fixed-point ProcessModels to be used must be
        specificed via setting the run configurations.
        After conversion has been executed, the converted parameters are stored
        in the member 'scaled_params' of the class, a dictionary with keys
        Process id and Variable name.

        The conversion has the following steps:
            1. Getting the Processes that need to be converted.
            2. Fetching all Vars and OutPorts and RefPorts of the Processes.
            3. If there is a hierarchical Process among the Processes to be
               converted, instantiate new Processes copying the structure of
               the passed model without hierarchical Processes. In the
               following, the conversion will be performed on the new model.
               Redo step 2 with the new model and continue with 4.
            4. Finding and instantiating the fixed-point ProcessModels.
            5. Retrieving the information needed for conversion from the
               fixed-point ProcessModels chosen via the provided fixed-point
               run configuration.
            6. Creating monitors for recording the activation statistics of the
               dynamical variables of Processes. With this, the domain of the
               variable can be determined to constrain the floating- to
               fixed-point conversion.
            7. Execution of the floating-point model chosen via run
               configuration. If quantile information is passed, the domain of
               the dynamical variables will in general be smaller than taking
               just the minimum and maximum of the recorded data. The default
               value ((0, 1)) will not lead to shrinkage of domain.
            8. Updating the scale_domains, i.e. the collection of variables
               that need to be scaled identically.
            9. Finally, scale the parameters so that no over/underflow occurs
               given the variability of the variables recorded in step 4.

        Parameters
        ----------

        proc : AbstracProcess or List[AbstractProcess]
            Processes to be converted
        num_steps : int
            Number of steps the floating-point model is executed for. A larger
            number of steps will lead to better statistics of the monitored
            variables which is needed for the conversion but also will lead to
            a longer run time
        quantiles : tuple
            Tuple of two numbers both in the interval [0,1] determining the
            p-quantile of the domain of a monitored variable to set largest and
            smallest value of the dynamic rep_range of that variable. The
            dynamic range will be used to determine the optimal floating- to
            fixed-point conversion. The first entry must be smaller than the
            second.
        find_connected_procs : bool
            Find all Processes connected to passed Process
        """
        self.num_steps = num_steps
        self.quantiles = quantiles
        self._set_procs(proc=proc, find_connected_procs=find_connected_procs)
        # Infer Variables and Ports of Processes.
        self.var_ports = self._get_var_ports()
        # If hierarchical Processes are involved, set up copy of model where
        # Subprocesses are made explicit.
        if self.hierarchical_procs:
            self._explicate_hierarchical_procs()
        else:
            pass
        # Get fixed-point ProcessModels of Processes.
        self.fixed_pt_proc_models = self._get_fixed_pt_proc_models()
        # Fetch conversion data from fixed-point ProcessModel.
        self.conv_data = self._get_conv_data()
        # Instantiate monitors for recording from dynamic variables.
        self.monitors = self._create_monitoring_infrastructure()
        # Run floating-point model and store activations monitored variables.
        self._run_procs()
        # Update scale domains for scaling of parameters.
        self.scale_domains = self._update_scale_domains()
        # Get the factor/ for the scaling functions for parameter conversion.
        self.scaling_factors = self._find_scaling_factors()
        # Scale parameter from a floating- to a fixed-point representation.
        self.scaled_params = self._scale_parameters()

    def plot_var(self, p_id, var, fig=None, bins='auto') -> Figure:
        """Plot histogram of Variable values of Process after execution of
        floating-point model in `fig'. Additionally, quantiles capping the
        recored data for further processing are plotted in red.
        The histogram is determined via `numpy.histogram'.
        This function can only be executed after calling `convert'.

        Parameters
        ----------
        p_id : int
            ProcessID
        var : string
            Name of Variable
        fig : matplotlib.figure.Figure
        bins : int or string
            Number of equal-width bins if bins is int
        """
        # Set up figure.
        if not fig:
            fig = plt.figure()

        ax = fig.gca()

        # Get data and calculate quantiles.
        data = self.conv_data[p_id][var]['domain']
        lower_val = np.quantile(data, self.quantiles[0])
        higher_val = np.quantile(data, self.quantiles[1])

        # Plot data.
        ax.set_title(f'Process: {p_id}, Variable: ' + var)
        ax.hist(data, bins=bins)
        ax.axvline(lower_val, color='red')
        ax.axvline(higher_val, color='red')

        return fig

    def _set_procs(self, proc, find_connected_procs=True) -> None:
        """Set list of Lava Processes for conversion. Either pass a Lava
        Process and find connected Lava Processes or pass list of Lava
        Processes.
        Write Processes p in a dictionary with structure {p.id : p}.

        Parameters
        ----------

        proc : AbstracProcess or List[AbstractProcess]
            Processes to be converted
        find_connected_procs : bool
            Find all Processes connected to passed Process
        """
        if issubclass(type(proc), AbstractProcess):
            if find_connected_procs:
                proc_list = find_processes(proc)
            else:
                proc_list = [proc]
        elif isinstance(proc, list):
            for p in proc:
                if not issubclass(type(p), AbstractProcess):
                    raise TypeError("list 'proc' must contain Processes."
                                    + f" 'proc' contains {p} with type"
                                    + f" {type(p)}.")
                proc_list = proc
        else:
            raise TypeError("'proc' must be Process of list of Processes."
                            + f"'proc' {proc} and has type {type(proc)}")

        # Get ProcessModel classes of Processes with fixed-point RunConfig.
        p_model_cls = ProcGroupDiGraphs._map_proc_to_model(
            proc_list,
            self.floating_pt_run_cfg)

        # Instantiate fixed-point ProcessModel of Processes and put them into
        # dictionary. Moreover we update the self.procs from a new proc list
        # containing Subprocesses but not their Hierarchical Processes anymore.
        proc_list_w_sub = list(p_model_cls.keys())

        # Find hierarchical Processes
        hierarchical_procs_set = set(proc_list) - set(proc_list_w_sub)

        # Updated Processes used for covnersion and set member storing
        # hierarchical Processes. Get rid of Processes that are to be ignored.
        self.procs = dict([(p.id, p) for p in proc_list_w_sub if type(p) not
                          in self.ignored_procs])
        self.hierarchical_procs = dict([(p.id, p) for p in
                                        hierarchical_procs_set if type(p) not
                                        in self.ignored_procs])

        self.run_proc = proc_list[0]

    def _explicate_hierarchical_procs(self):
        """Given a Model with hierarchical Processes, set up an equivalent
        model with only leaf Processes, i.e. not hierarchical Processes.
        This is needed for the subsequent monitoring of Process Variables and
        Conversion. The newly created Processes are instantiated with the same
        intial values as all (Sub)Processes in the original model."""

        # Create dictionary for storing the conenction date.
        conn_data = {}

        for p_id, p_var_ports in self.var_ports.items():
            # Increase depth of dictionary to collect connection data.
            conn_data[p_id] = {}

            p = self.procs[p_id]

            for out in p_var_ports['SourcePort']:
                # Get target Ports as list.
                target_ports = p.__getattribute__(out).out_connections

                target_ports = self._update_target_ports(target_ports)

                # Store connection information.
                conn_data[p_id][out] = target_ports

        # Store newly created Procceses for later use.
        procs_new = {}
        # Keep track of Process Ids between orginal and equivalend Model
        p_id_pairs = {}

        # Build model from scratch from first order processes.
        for p_id, p in self.procs.items():
            type_p = type(p)
            # Get inital values for Process p.
            p_params = dict(inspect.signature(p.__init__).parameters)
            p_params.pop('name', None)
            p_params.pop('log_config', None)

            # Add keyword arguments for instantiating new Procs.
            kwargs = {}

            for param_name in p_params.keys():
                try:
                    param = p.__getattribute__(param_name)
                except AttributeError:
                    continue

                if type(param) is Var:
                    # Get initial condition.
                    param_value = param.init
                else:
                    param_value = param

                kwargs[param_name] = param_value

            # Instantiate Process p_new.
            p_new = type_p(**kwargs)

            # Update procs.
            procs_new[p_new.id] = p_new

            p_id_pairs[p_id] = p_new.id

        # Connect newly instantiated processes.
        for p_id, out_ports in conn_data.items():
            p = procs_new[p_id_pairs[p_id]]
            for out, target_ports in out_ports.items():
                p_op = p.__getattribute__(out)
                for t_port in target_ports:
                    t_port_p_id = t_port.process.id
                    t_port_name = t_port.name

                    t_p = procs_new[p_id_pairs[t_port_p_id]]

                    t_port = t_p.__getattribute__(t_port_name)

                    p_op.connect(t_port)

        self.procs = procs_new
        self.run_proc = p_new

        # Fetch Variables and SourcePorts for new model.
        self.var_ports = self._get_var_ports()

    def _get_var_ports(self) -> dict:
        """Get Vars, Source Ports (including OutPorts and RefPorts of Processes
        in converter and returns a nested dictionary with keys Process ID, keys
        Vars/Source Port, and items the name of Vars/Source Ports in a set.

        Returns
        -------
        vars_ports : dict
            Dictionary containing names of Vars and Ports of Processes
        """
        var_ports = {}

        for p_id, p in self.procs.items():
            # Initialize inner dictionaries.
            var_ports[p.id] = {}

            # Get Vars, Out/RefPorts of Process p.
            p_dict = p.__dict__
            p_var = [key for key in p_dict.keys() if type(p_dict[key]) is Var]
            p_s = [key for key in p_dict.keys()
                   if issubclass(type(p_dict[key]), AbstractSrcPort)]

            var_ports[p_id]['Var'] = set(p_var)
            var_ports[p_id]['SourcePort'] = set(p_s)

        return var_ports

    def _get_fixed_pt_proc_models(self) -> dict:
        """Get fixed-point ProcessModels of passed Processes and returns
        dictionary with keys Process ID and items ProcessModels.

        Returns
        -------
        fixed_pt_proc_models : dict
            Dictionary containing the ProcessModels for Processes
        """
        # Get ProcessModel classes of Processes with fixed-point RunConfig.
        p_model_cls = ProcGroupDiGraphs._map_proc_to_model(
            list(self.procs.values()),
            self.fixed_pt_run_cfg)

        # Instantiate fixed-point ProcessModel of Processes and put them into
        # dictionary. Moreover we update the self.procs from a new proc list
        # containing Subprocesses but not their Hierarchical Processes anymore.
        fixed_pt_proc_models = {}
        proc_list = []
        for p, model_cls in p_model_cls.items():
            fixed_pt_proc_models[p.id] = model_cls(p.proc_params)
            proc_list.append(p)

        return fixed_pt_proc_models

    def _get_conv_data(self) -> dict:
        """Get and update the conversion data of Vars from Processes that need
        to be converted. Data is fetched from LavaPyType of Vars of fixed-point
        process models specified via fixed-point run config.
        This is the precision information, a-priori known domains, whether
        a Var is constant throughout execution, and, if applicable, the number
        of bits for the exponent.
        Updating includes modifying implicit shifts in the precision
        information due to different scales between process and a check for
        consistency for these scales.
        Conversion data is stored in nested dictionary with key Process Id, key
        Variable Name key precision, domain, constant, and num_exp_bit.

        Returns
        -------
        conv_data : dict
            Conversion data for Float2Fixed converter in nested dict.
            {ProcID:
                {VarName:
                    {'precision': _,
                     'domain': _,
                     'constant: _,
                     'num_bits_exp':_}}}
        """
        # Create dictionary to store conversion data.
        conv_data = {}

        for p_id, p_var_ports in self.var_ports.items():
            # Increase depth of dictionary to collect information about Vars of
            # Process and connections.
            conv_data[p_id] = {}
            # Get Process
            p = self.procs[p_id]
            # Get ProcessModel.
            p_model = self.fixed_pt_proc_models[p_id]

            # Check whether process has out connections that are scaled.
            # If so, check for consistency of this scaling
            p_impl_shift = False
            for out in p_var_ports['SourcePort']:
                # Get target Ports as list.
                target_ports = p.__getattribute__(out).out_connections

                # Get if port is OutPort
                outport = (type(p.__getattribute__(out)) is OutPort)

                # Update target Ports. If target Port is Port of hierarchcical
                # Procass find next Prots of Subprocesses.
                target_ports = self._update_target_ports(target_ports)

                for t_port in target_ports:

                    # If Outport get name of target Port.
                    if outport:
                        t_name = t_port.name
                    # Else get name of variable targeted by RefPort
                    else:
                        t_name = t_port.var.name

                    # Get id of Process hosting target Port.
                    t_port_p_id = t_port.process.id
                    # Get ProcessModel of Process hosting target Port.
                    t_port_p_model = self.fixed_pt_proc_models[t_port_p_id]
                    # Get LavaPyType of target Port.
                    t_port_py_type = t_port_p_model.__getattribute__(t_name)
                    # Get precision information of target Port
                    try:
                        t_port_conv_data = t_port_py_type.conversion_data()
                    except (TypeError, ValueError, AttributeError):
                        raise ValueError(f'Error in {t_port} of Proc {p_id}')
                    t_port_impl_shift = t_port_conv_data['implicit_shift']
                    # If there has been not implicit shift set so far set one.
                    if not p_impl_shift:
                        p_impl_shift = t_port_impl_shift

                    # Check for consistency of implicit shifts due to Ports
                    elif not p_impl_shift == t_port_impl_shift:
                        raise ValueError(
                            "Implicit shift due to connections inconsistent in"
                            + f" Process {p_id}. Cannot perform conversion.")

            for var in p_var_ports['Var']:
                # Get LavaPyType instance of var.
                var_py_type = p_model.__getattribute__(var)
                # Skip meta parameters since they don't need to be converted.
                if var_py_type.meta_parameter:
                    continue

                # Retrieve conversion data.
                try:
                    conv_data[p_id][var] = var_py_type.conversion_data()
                except (TypeError, ValueError, AttributeError):
                    raise ValueError(f'Error in {var} of Proc {p_id}')

                # Update implicit shift if Variable is on global scale domain.
                if conv_data[p_id][var]['scale_domain'] == 0:
                    conv_data[p_id][var]['implicit_shift'] += p_impl_shift

                # If Variable var is constant and domain is None store initial
                # value in domain.
                update_domain = (
                    conv_data[p_id][var]['constant']
                    and conv_data[p_id][var]['domain'] is None
                )
                if update_domain:
                    init = self.procs[p_id].__getattribute__(var).init
                    conv_data[p_id][var]['domain'] = init

                # Keep track of which scale domain has variables in Processes.
                self.scale_domains.add(
                    (conv_data[p_id][var]['scale_domain'], p_id))

        return conv_data

    def _create_monitoring_infrastructure(self) -> dict:
        """Create and connect monitors for determining the dynamic range of
        variables in the model that needs to be converted.
        Only variables which are neither a meta parameter nor a constant nor
        have a predefined dynamic range will be recorded.
        Monitors are stored in a nested dictionary with key Process Id, key
        Variable name.

        Returns
        -------
        monitors : dict
            Monitoring infrastructure for Float2Fixed converter in nested dict
            {ProcID:
                {VarName:
                     monitor}}
        """
        monitors = {}

        for p_id, conv_data_p_id in self.conv_data.items():
            monitors[p_id] = {}
            for var, conv_data in conv_data_p_id.items():

                # No monitoring if var not dynamic.
                if (conv_data['constant'] or conv_data['domain'] is not None):
                    continue

                var_obj = self.procs[p_id].__getattribute__(var)

                # Instantiate, connect and store monitor.
                m = Monitor()
                m.probe(target=var_obj,
                        num_steps=self.num_steps)

                monitors[p_id][var] = m

        return monitors

    def _run_procs(self) -> None:
        """Run Processes passed to converter with given number of steps, get
        and store data of monitored Variables in conversion data dictionary.
        The needed 'run_config' is taken from the passed 'floating_pt_run_cfg'.
        """
        run_cfg = self.floating_pt_run_cfg
        run_cond = RunSteps(num_steps=self.num_steps)

        # Run Processes from designated Process.
        self.run_proc.run(condition=run_cond, run_cfg=run_cfg)
        # Get monitored variables.
        for p_id, monitor_dict in self.monitors.items():
            p = self.procs[p_id]
            for var, monitor in monitor_dict.items():
                var_obj = p.__getattribute__(var)
                data = monitor.get_data()[p.name][var_obj.name].flatten()
                # Store data as 'domain' in 'conv_data'.
                self.conv_data[p_id][var]['domain'] = data

        self.run_proc.stop()

    def _update_scale_domains(self) -> dict:
        """Update scale domains by Separate and store the information for the
        float- to fixed-point conversion according to the scale domains in a
        dictionary of dictionaries with keys scale domain, Process id,
        Variable name, conversion data. Different list entries correspond to a
        set of variables that need to be converted consistently.

        Returns
        -------
        scale_domains : dict
            Conversion data for Float2Fixed converter in nested dict.
            {ScaleDomain:
                {ProcID:
                   {VarName:
                        {'precision': _,
                         'domain': _,
                         'constant: _,
                         'num_bits_exp':_}}}}
        """

        # Initialize scale domain dictionary.
        scale_domains = {}
        for scale_domain, p_id in self.scale_domains:
            # Only create entry if scale_domain has now entry in dict yet.
            if scale_domain not in scale_domains.keys():
                scale_domains[scale_domain] = {}
            scale_domains[scale_domain][p_id] = {}

        for p_id, conv_data_p_id in self.conv_data.items():
            for var, conv_data in conv_data_p_id.items():
                # Create copy of conv_data.
                conv_data_copy = conv_data.copy()

                # Get scale domain and remove it from dictionary.
                scale_domain = conv_data_copy.pop('scale_domain')
                # Remove other information not needed anymore.
                conv_data_copy.pop('constant')
                conv_data_copy.pop('exp_var')

                if conv_data_copy['num_bits_exp'] is None:
                    conv_data_copy['num_bits_exp'] = 0
                scale_domains[scale_domain][p_id][var] = conv_data_copy

        return scale_domains

    def _find_scaling_factors(self) -> dict:
        """Determine the optimal scaling function for the float- to fixed-point
        mapping. The function must be consistent between Variables of one scale
        domain of be chosen such that for every Variable the numerical
        fixed-point representation can be represented with the given
        constraints. The optimal scaling function will be stored in a
        dictionary with keys scale domains.

        Returns
        -------
        scaling_funct_dict: dict
            Dictionary containing the optimal scaling function for each scale
            domain.
        """
        # Initialize scaling function dictionary.
        scaling_factors = {}
        slopes = []

        for scale_domain, scale_data_domain in self.scale_domains.items():
            scaling_factors[scale_domain] = {}
            for p_id, scale_data_p_id in scale_data_domain.items():
                for var, scale_data in scale_data_p_id.items():

                    bits = scale_data['num_bits']
                    impl_shift = scale_data['implicit_shift']
                    num_bits_exp = scale_data['num_bits_exp']

                    # Get range representable given bit constraints and whether
                    # representation is signed..
                    if scale_data['is_signed']:
                        rep_range = np.array([-1 * 2 ** (bits - 1),
                                              2 ** (bits - 1) - 1])
                    else:
                        rep_range = np.array([0, 2 ** bits - 1])

                    # Apply implicit shift.
                    rep_range = np.left_shift(rep_range, impl_shift)

                    # Apply maximal shift due to exponent.
                    rep_range = np.left_shift(rep_range, 2 ** num_bits_exp - 1)

                    # Get range of Variables in floating point domain.
                    domain = scale_data['domain']

                    # If domain of Variable has been recorded apply quantile.
                    if var in self.monitors[p_id].keys():
                        lower_val = np.quantile(domain, self.quantiles[0])
                        higher_val = np.quantile(domain, self.quantiles[1])
                        dyn_range = (float(lower_val), float(higher_val))
                    else:
                        dyn_range = (float(np.min(domain)),
                                     float(np.max(domain)))

                    # Calculate slopes for linear mapping of var.
                    # If an entry in rep_range or dyn_range is zero, inf will
                    # be filled in.
                    dyn_range_arr = np.array(dyn_range)
                    not_zero_cond = ((rep_range != 0) * (dyn_range_arr != 0))
                    slopes_var = np.divide(rep_range, dyn_range_arr,
                                           out=(np.inf
                                                * np.ones_like(dyn_range_arr)),
                                           where=not_zero_cond)

                    # Take absolute value. Only positive slopes are allowed.
                    slopes_var = np.absolute(slopes_var)

                    # Choose smallest slope to avoid scaling inconsistencies.
                    slopes.append(np.min(slopes_var))

                # If scale domain not zero, scaling is Process specific.
                if scale_domain:
                    m = np.min(slopes)
                    scaling_factors[scale_domain][p_id] = m.copy()

                    slopes.clear()

            # If scale domain zero, scaling is global.
            if not scale_domain:
                m = np.min(slopes)
                scaling_factors[scale_domain] = m.copy()

                slopes.clear()

        return scaling_factors

    def _scale_parameters(self) -> dict:
        """Scale the initial values of the passed Processes given the slopes
        for the scale domain in 'scaling_factors' from a floating-point to
        a fixed-point representation. The parameters are stored in a nested
        dictionary with keys Process id and Variable name.

        Returns
        -------
        scaled_params : dict
            Dictionary containing parameters scaled to fixed-point
            representation with keys ProcessID, Variable name
        """
        scaled_params = dict([(p_id, {}) for p_id in self.procs.keys()])

        # Meta parameters must be added separately
        for p_id, p_var_ports in self.var_ports.items():
            for var in p_var_ports['Var']:
                if var not in self.conv_data[p_id].keys():
                    meta_param = self.procs[p_id].__getattribute__(var).init
                    scaled_params[p_id][var] = meta_param

        # Transform parameters to fixed-point domain and store them.
        for scale_domain, scale_data_domain in self.scale_domains.items():
            for p_id, scale_data_p_id in scale_data_domain.items():
                if scale_domain == 0:
                    scaling_factor = self.scaling_factors[scale_domain]
                else:
                    scaling_factor = self.scaling_factors[scale_domain][p_id]

                for var, scale_data in scale_data_p_id.items():
                    # Get information needed for conversion of floating to
                    # fixed-point numbers.
                    param = self.procs[p_id].__getattribute__(var).init
                    impl_shift = scale_data['implicit_shift']
                    num_exp_bits = scale_data['num_bits_exp']
                    # Apply implicit shift and convert floating- to fixed-point
                    # representation.
                    scaled_param = np.round(scaling_factor * param
                                            / 2 ** impl_shift).astype(int)

                    if not num_exp_bits:
                        scaled_params[p_id][var] = scaled_param
                    else:
                        # Calculate mantissa and exponent and store them
                        # separately.
                        exp_var = self.conv_data[p_id][var]['exp_var']
                        if scale_data['is_signed']:
                            sign_bit = 1
                        else:
                            sign_bit = 0

                        eff_mantissa_bits = (scale_data['num_bits']
                                             - sign_bit)

                        if np.any(scaled_param):
                            exp = (np.log2(np.max(np.abs(scaled_param)))
                                   - eff_mantissa_bits)
                            exp = np.ceil(exp).astype(int)
                        else:
                            exp = 0
                        exp = np.max([exp, 0])

                        scaled_param = np.right_shift(scaled_param, exp)

                        scaled_params[p_id][var] = scaled_param
                        scaled_params[p_id][exp_var] = exp

        return scaled_params

    def _update_target_ports(self, target_ports: list) -> list:
        """Recursively update target Port of OutPort. If target Port is Port of
        hierarchical Process trace Ports to first (Sub)Process that is
        connected.

        Parameters
        ----------
        target_ports : list
            List of target ports of OutPort

        Retunrs
        -------
        updated_target_ports : list
            List of target Ports where all Ports belong the (Sub)Processes
        """
        updated_target_ports = []
        for t_port in target_ports:

            t_port_p_id = t_port.process.id

            if t_port_p_id in self.procs.keys():
                updated_target_ports.extend([t_port])

            # Does target Port Process id belong to a hierarchical Process?
            elif t_port_p_id in self.hierarchical_procs.keys():
                updated_target_ports.extend(
                    self._update_target_ports(t_port.out_connections))

        return updated_target_ports
