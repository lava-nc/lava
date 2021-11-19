# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort

class Monitor(AbstractProcess):
    """
    Monitor process to probe/monitor a given variable of a process

    Monitor process is initialized without any ports and Vars. The InPorts,
    RefPorts and Vars are created dynamically, as the Monitor process is
    used to probe OutPorts and Vars of other processes. For this purpose,
    Monitor process has probe(..) function, which as arguments takes the
    target Var or OutPorts and number of time steps we want to monitor given
    process.
    multiple instance variables

    Attributes
    ----------
    data : dict
        Dictionary that is populated by monitoring data once get_data(..)
        method is called, has the follwoing structure:
        data
          \__Monitored_process_name
             \__Monitored_var_or_outPort_name

    proc_params: dict
        Process parameters that will be transferred to the corresponding
        ProcessModel. It is populated with the names of dynamically
        created port and var names of Monitor process, to be carried to its
        ProcessModel. It is a dictionary of the following structure:
            "RefPorts": names of RefPorts created to monitor target Vars
            "VarsRead": names of Vars created to store data from target Vars
            "InPorts": names of InPorts created to monitor target OutPorts
            "OutsRead": names of Vars created to store data from target OutPorts
            "n_ref_ports": number of created RefPorts, also monitored Vars
            "n_in_ports": number of created InPorts, also monitored OutPorts

    target_names: dict
        The names of the targeted Processes and Vars/OutPorts to be monitored.
        This is used in get_data(..) method to access the target names
        corresponding to data-storing Vars of Monitor process during readout
        phase. This dict has the follwoing sturcture:
            key: name of the data-storing Vars, i.e. VarsRead and OutsRead
            value: [Monitored_process_name, Monitored_var_or_outPort_name]

    Methods
    -------
    post_init()
        Create one prototypical RefPort, InPort and two Vars. This ensure
        coherence and one-to-one correspondence between Monitor process and
        ProcessModel in terms LavaPyTypes and Ports/Vars. These prototypical
        ports can later be updated inside probe(..) method.

    probe(target, num_steps)
        Probe the given target for num_step time steps, where target can be
        a Var or OutPort of some process.

    get_data()
        Fetch the monitoring data from the Vars of Monitor process that
        collected it during the run from probed process, puts into dict form
        for easier access by user
    """
    def __init__(self, **kwargs):
        """
        Initialize the attributes and run post()
        """
        super().__init__(**kwargs)
        # self.ref = []
        self.data = {}

        self.proc_params["RefPorts"] = []
        self.proc_params["VarsRead"] = []
        self.proc_params["InPorts"] = []
        self.proc_params["OutsRead"] = []
        self.proc_params["n_ref_ports"] = 0
        self.proc_params["n_in_ports"] = 0
        
        self.target_names = {}

        self.post_init()

    def post_init(self):
        """
        Create one prototypical RefPort, InPort and two Vars. This ensure
        coherence and one-to-one correspondence between Monitor process and
        ProcessModel in terms LavaPyTypes and Ports/Vars. These prototypical
        ports can later be updated inside probe(..) method.
        Note: This is separated from constructor, because once
        multi-variable monitoring is enabled, this method will be depracated
        """

        # Create names for prototypical Ports/Vars to be created in Monitor
        # process for porbing purposes.
        self.new_ref_port_name = "ref_port_" + \
                                 str(self.proc_params["n_ref_ports"])
        self.new_var_read_name = "var_read_" + \
                                 str(self.proc_params["n_ref_ports"])
        self.new_in_port_name = "in_port_" + \
                                str(self.proc_params["n_in_ports"])
        self.new_out_read_name = "out_read_" + \
                                 str(self.proc_params["n_in_ports"])

        # create and set new Refport and corresponding Var to store data
        setattr(self, self.new_ref_port_name, RefPort(shape=(1,)))
        setattr(self, self.new_var_read_name, Var(shape=(1,), init=0))

        # create and set new InPort and corresponding Var to store data
        setattr(self, self.new_in_port_name, InPort(shape=(1,)))
        setattr(self, self.new_out_read_name,Var(shape=(1,), init=0))

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
        Probe the given target for num_step time steps, where target can be
        a Var or OutPort of some process.

        Parameters
        ----------
        target : Var or OutPort
            a Var or OutPort of some process to be monitored.
        num_steps: int
            The number of steps the target Var/OutPort should be monitored.
        """

        # Create names for Ports/Vars to be created in Monitor  process for
        # porbing purposes. Names are given incrementally each time proce(..)
        # method is called
        self.new_ref_port_name = "ref_port_" + \
                                 str(self.proc_params["n_ref_ports"])
        self.new_var_read_name = "var_read_" + \
                                 str(self.proc_params["n_ref_ports"])
        self.new_in_port_name = "in_port_" + \
                                str(self.proc_params["n_in_ports"])
        self.new_out_read_name = "out_read_" + \
                                 str(self.proc_params["n_in_ports"])

        # create and set new Refport and corresponding Var to store data
        setattr(self, self.new_ref_port_name, RefPort(shape=target.shape))
        setattr(self, self.new_var_read_name, Var(shape=(target.shape[0],
                                                    num_steps), init=0))

        # create and set new InPort and corresponding Var to store data
        setattr(self, self.new_in_port_name, InPort(shape=target.shape))
        setattr(self, self.new_out_read_name,
                Var(shape=(target.shape[0], num_steps),
                    init=0))

        # Add the names of new RefPort and Var_read name to proc_params dict
        self.proc_params["RefPorts"].append(self.new_ref_port_name)
        self.proc_params["VarsRead"].append(self.new_var_read_name)

        # Add the names of new RefPort and Var_read name to proc_params dict
        self.proc_params["InPorts"].append(self.new_in_port_name)
        self.proc_params["OutsRead"].append(self.new_out_read_name)

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

        # if target to be monitored is a Var
        if isinstance(target, Var):

            # update id for the next use of probe(..) method
            self.proc_params["n_ref_ports"] += 1

            # connect newly created Refport to the var to be monitored
            getattr(self, self.new_ref_port_name).connect_var(target)

            # Add the name of probed Var and its process to the target_names
            self.target_names[self.new_var_read_name] = [target.process.name,
                                                         target.name]
        # if target to be monitored is an OutPort
        elif isinstance(target, OutPort):

            # update id for the next use of probe(..) method
            self.proc_params["n_in_ports"] += 1

            # connect newly created InPort from the OutPort to be monitored
            getattr(self, self.new_in_port_name).connect_from(target)

            # Add the name of probed OutPort and its process to the target_names
            self.target_names[self.new_out_read_name] = [target.process.name,
                                                         target.name]

        # if target is an InPort raise a Type error, as monitoring InPorts is
        # not supported yet
        else:
            raise TypeError("Non-supported probe target: type {}"
                            .format(target))

        # Create correspding dict keys for monitored Var/OutPort and its process
        self.data[str(target.process.name)] = {}
        self.data[str(target.process.name)][str(target.name)] = 0
        
        return

    def get_data(self):
        """
        Fetch the monitoring data from the Vars of Monitor process that
        collected it during the run from probed process, puts into dict form
        for easier access by user

        Returns
        -------
        data : dict
            Data dictionary collected by Monitor Process
        """

        # Fetch data-storing Vars for OutPort monitoring
        if self.proc_params["n_in_ports"] > 0:
            for i in range(self.proc_params["n_in_ports"]):
                data_var_name = self.proc_params["OutsRead"][i]
                data_var = getattr(self, data_var_name)
                target_name = self.target_names[data_var_name]

                self.data[target_name[0]][target_name[1]] = data_var.get()

        # Fetch data-storing Vars for Var monitoring
        if self.proc_params["n_ref_ports"] > 0:
            for i in range(self.proc_params["n_ref_ports"]):

                data_var_name = self.proc_params["VarsRead"][i]
                data_var = getattr(self, data_var_name)
                target_name = self.target_names[data_var_name]

                self.data[target_name[0]][target_name[1]] = data_var.get()

        return self.data

