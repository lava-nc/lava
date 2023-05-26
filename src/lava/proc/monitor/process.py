# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort


class Monitor(AbstractProcess):
    """
    Monitor process to probe/monitor a given variable of a target process.

    Monitor process is initialized without any Ports and Vars. The InPorts,
    RefPorts and Vars are created dynamically, as the Monitor process is
    used to probe OutPorts and Vars of other processes. For this purpose,
    Monitor process has the `probe()` method, which as arguments takes the
    target Var or OutPorts and number of time steps we want to monitor given
    process.

    Note: Monitor currently only supports to record from a singe Var or Port.

    Parameters
    ----------
    data : dict
        Dictionary that is populated by monitoring data once get_data(..)
        method is called, has the following structure:
        data
        __monitored_process_name
        __monitored_var_or_out_port_name

    proc_params: dict
        Process parameters that will be transferred to the corresponding
        ProcessModel. It is populated with the names of dynamically
        created port and var names of Monitor process, to be carried to its
        ProcessModel. It is a dictionary of the following structure:
        "RefPorts": names of RefPorts created to monitor target Vars
        "VarsData1": names of Vars created to store data from target Vars
        "InPorts": names of InPorts created to monitor target OutPorts
        "VarsData2": names of Vars created to store data from target OutPorts
        "n_ref_ports": number of created RefPorts, also monitored Vars
        "n_in_ports": number of created InPorts, also monitored OutPorts

    target_names: dict
        The names of the targeted Processes and Vars/OutPorts to be monitored.
        This is used in get_data(..) method to access the target names
        corresponding to data-storing Vars of Monitor process during readout
        phase. This dict has the follwoing sturcture:
        key: name of the data-storing Vars, i.e. VarsData1 and VarsData2
        value: [monitored_process_name, monitored_var_or_out_port_name]

    """

    def __init__(self):
        """Initialize the attributes and run post_init()."""
        super().__init__()

        self.data = {}

        self.proc_params["RefPorts"] = []
        self.proc_params["VarsData1"] = []
        self.proc_params["InPorts"] = []
        self.proc_params["VarsData2"] = []
        self.proc_params["n_ref_ports"] = 0
        self.proc_params["n_in_ports"] = 0

        self.target_names = {}

        self.post_init()

    def post_init(self):
        """
        Run after __init__.

        Creates one prototypical RefPort, InPort and two Vars.
        This ensures coherence and one-to-one correspondence between the
        Monitor Process and ProcessModel in terms og LavaPyTypes and
        Ports/Vars. These prototypical ports can later be updated inside the
        `probe()` method.

        Note: This is separated from constructor, because once multi-variable
        monitoring is enabled, this method will be deprecated.
        """
        # Create names for prototypical Ports/Vars to be created in Monitor
        # process for probing purposes.
        self.new_ref_port_name = "ref_port_" + \
                                 str(self.proc_params["n_ref_ports"])
        self.new_var_read_name = "var_read_" + \
                                 str(self.proc_params["n_ref_ports"])
        self.new_in_port_name = "in_port_" + \
                                str(self.proc_params["n_in_ports"])
        self.new_out_read_name = "out_read_" + \
                                 str(self.proc_params["n_in_ports"])

        # Create and set new Refport and corresponding Var to store data
        setattr(self, self.new_ref_port_name, RefPort(shape=(1,)))
        setattr(self, self.new_var_read_name, Var(shape=(1,), init=0))

        # Create and set new InPort and corresponding Var to store data
        setattr(self, self.new_in_port_name, InPort(shape=(1,)))
        setattr(self, self.new_out_read_name, Var(shape=(1,), init=0))

        # Register new created Vars/Ports
        attrs = self._find_attr_by_type(RefPort)
        self._init_proc_member_obj(attrs)
        self.ref_ports.add_members(attrs)

        attrs = self._find_attr_by_type(Var)
        self._init_proc_member_obj(attrs)
        self.vars.add_members(attrs)

        attrs = self._find_attr_by_type(InPort)
        self._init_proc_member_obj(attrs)
        self.in_ports.add_members(attrs)

    def probe(self, target, num_steps):
        """
        Probe a Var or OutPort to record data from.

        Record the target for num_step time steps, where target can be
        a Var or OutPort of a process.

        Parameters
        ----------
        target : Var or OutPort
            a Var or OutPort of some process to be monitored.
        num_steps: int
            The number of steps the target Var/OutPort should be monitored.
        """
        # Create names for Ports/Vars to be created in Monitor process for
        # probing purposes. Names are given incrementally each time probe(..)
        # method is called.
        new_ref_port_name = f"ref_port_{self.proc_params['n_ref_ports']}"
        new_var_read_name = f"var_read_{self.proc_params['n_ref_ports']}"
        new_in_port_name = f"in_port_{self.proc_params['n_in_ports']}"
        new_out_read_name = f"out_read_{self.proc_params['n_in_ports']}"

        # Create and set new Refport and corresponding Var to store data
        setattr(self, new_ref_port_name, RefPort(shape=target.shape))
        setattr(self, new_var_read_name,
                Var(shape=(num_steps,) + target.shape, init=0))

        # Create and set new InPort and corresponding Var to store data
        setattr(self, new_in_port_name, InPort(shape=target.shape))
        setattr(self, new_out_read_name,
                Var(shape=(num_steps,) + target.shape, init=0))

        # Add the names of new RefPort and Var_read name to proc_params dict
        self.proc_params["RefPorts"].append(self.new_ref_port_name)
        self.proc_params["VarsData1"].append(self.new_var_read_name)

        # Add the names of new RefPort and Var_read name to proc_params dict
        self.proc_params["InPorts"].append(self.new_in_port_name)
        self.proc_params["VarsData2"].append(self.new_out_read_name)

        # Register new created Vars/Ports
        attrs = self._find_attr_by_type(RefPort)
        self._init_proc_member_obj(attrs)
        self.ref_ports.add_members(attrs)

        attrs = self._find_attr_by_type(Var)
        self._init_proc_member_obj(attrs)
        self.vars.add_members(attrs)

        attrs = self._find_attr_by_type(InPort)
        self._init_proc_member_obj(attrs)
        self.in_ports.add_members(attrs)

        # If target to be monitored is a Var
        if isinstance(target, Var):

            # Update id for the next use of probe(..) method
            n_ref_ports = self.proc_params["n_ref_ports"]
            self.proc_params.overwrite("n_ref_ports", n_ref_ports + 1)

            # Connect newly created Refport to the var to be monitored
            getattr(self, new_ref_port_name).connect_var(target)

            # Add the name of probed Var and its process to the target_names
            self.target_names[new_var_read_name] = [target.process.name,
                                                    target.name]
        # If target to be monitored is an OutPort
        elif isinstance(target, OutPort):

            # Update id for the next use of probe(..) method
            n_in_ports = self.proc_params["n_in_ports"]
            self.proc_params.overwrite("n_in_ports", n_in_ports + 1)

            # Connect newly created InPort from the OutPort to be monitored
            getattr(self, new_in_port_name).connect_from(target)

            # Add the name of OutPort and its process to the target_names
            self.target_names[new_out_read_name] = [target.process.name,
                                                    target.name]

        # If target is an InPort raise a Type error, as monitoring InPorts is
        # not supported yet
        else:
            raise TypeError("Non-supported probe target: type {}"
                            .format(target))

        # Create corresponding dict keys for monitored Var/OutPort and its
        # process
        self.data[str(target.process.name)] = {}
        self.data[str(target.process.name)][str(target.name)] = 0

        return

    def get_data(self):
        """
        Fetch and return the recorded data.

        The recorded data is fetched, presented in a readable
        dict format and returned.

        Returns
        -------
        data : dict
            Data dictionary collected by Monitor Process.
        """
        # Fetch data-storing Vars for OutPort monitoring
        for i in range(self.proc_params["n_in_ports"]):
            data_var_name = self.proc_params["VarsData2"][i]
            data_var = getattr(self, data_var_name)
            target_name = self.target_names[data_var_name]

            self.data[target_name[0]][target_name[1]] = data_var.get()

        # Fetch data-storing Vars for Var monitoring
        for i in range(self.proc_params["n_ref_ports"]):
            data_var_name = self.proc_params["VarsData1"][i]
            data_var = getattr(self, data_var_name)
            target_name = self.target_names[data_var_name]

            self.data[target_name[0]][target_name[1]] = data_var.get()

        return self.data

    def plot(self, ax, target, *args, **kwargs):
        """
        Plot the recorded data into subplots.

        Can handle recordings of multiple processes and multiple variables
        per process.
        Each process will create a separate column in the subplots, each
        variable will be plotted in a separate row.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes to plot the data into
        target: Var or OutPort
            The target which should be plotted
        *args
            Passed to the matplotlib.plot function to customize the plot.
        **kwargs
            Passed to the matplotlib.plot function to customize the plot.
        """

        # fetch data
        data = self.get_data()
        # set plot attributes
        ax.set_title(target.process.name)
        ax.set_xlabel("Time step")
        ax.set_ylabel(target.name)

        # plot data
        ax.plot(data[target.process.name][target.name], *args, **kwargs)
