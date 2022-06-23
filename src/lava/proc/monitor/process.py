# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort


class Monitor(AbstractProcess):
    """
    Monitor process to probe/monitor target variables (Vars) or OutPorts of
    the other processes

    Monitor process uses RefPorts and InPorts to monitor the Vars and
    OutPorts of other processes, respectively. It also uses (memory) Vars to
    store the collected data during monitoring. The InPorts, RefPorts and
    Vars are created dynamically, as user calls probe(..) function. This
    function takes the target Var or OutPorts and number of time steps we
    want to monitor given process, as arguments. Subsequently user can
    extract the collected data from memory Vars of Monitor process using
    get_data(..) function.

    Attributes
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
          "ref_ports": names of RefPorts created to monitor target Vars
          "mem_vars_rp": names of Vars created to store data from target Vars
          "in_ports": names of InPorts created to monitor target OutPorts
          "mem_vars_ip": names of Vars created to store data from target
          OutPorts
          "n_ref_ports": number of created RefPorts, also monitored Vars
          "n_in_ports": number of created InPorts, also monitored OutPorts

    target_names: dict
        The names of the targeted Processes and Vars/OutPorts to be monitored.
        This is used in get_data(..) method to access the target names
        corresponding to data-storing Vars of Monitor process during readout
        phase. This dict has the following structure:
            key: name of the data-storing Vars, i.e. VarsData1 and VarsData2
            value: [monitored_process_name, monitored_var_or_out_port_name]

    Methods
    -------

    probe(target, num_steps)
        Probe the given target for num_step time steps, where target can be
        a Var or OutPort of some process.
    create_ref_port_and_mem_var(shape, num_steps)
        Create a new RefPort and (memory) Var inside the process
    create_in_port_and_mem_var(shape, num_steps)
        Create a new InPort and (memory) Var inside the process
    get_data()
        Fetch the monitoring data from the Vars of Monitor process that
        collected it during the run from probed process, puts into dict form
        for easier access by user
    """

    def __init__(self, **kwargs):
        """
        Initializes the attributes and run post().
        """
        super().__init__(**kwargs)

        self.data = {}

        self.proc_params["ref_ports"] = []
        self.proc_params["mem_vars_rp"] = []
        self.proc_params["in_ports"] = []
        self.proc_params["mem_vars_ip"] = []

        self.proc_params["n_ref_ports"] = 0
        self.proc_params["n_in_ports"] = 0

        self.target_names = {}

        # Contaiiner names for RefPorts, InPorts and (memory) Vars
        self.ref_port_cont_nm = "ref_ports_list"
        self.mem_var_for_rp_cont_nm = "mem_vars_rp_list"
        self.in_port_cont_nm = "in_ports_list"
        self.mem_var_for_ip_cont_nm = "mem_vars_ip_list"

        # Create dummy RefPort and InPort as part of initialization, so that
        # proc_model LavaType checks pass even if Monitor is not used to
        # probe only one type of target (Var or OutPort)
        self.create_ref_port_and_mem_var((1,), 1)
        self.create_in_port_and_mem_var((1,), 1)

    def create_ref_port_and_mem_var(self, shape, num_steps):
        """
        Create a new RefPort and (memory) Var inside the process. It assigns
        corresponding container names from the Monitor instance.
        Note: The number of RefPorts (proc_params["n_ref_ports"]) is not
        updated here, but rather inside the probe function, because in some
        cases it might be desirable to create the port and later override them.

        Parameters
        ----------
        shape : tuple
            The shape of RefPort will be created. This is used to
            determine the shape of Var required to store probed data

        num_steps : int
            Number of steps the (memory) Var will record the target,
            this is used as last dimension of the shape of Var

        Returns
        -------

        """
        # Create names for RefPorts and Vars to be created in Monitor process
        # for Var probing purposes. Names are given incrementally each time
        # probe(..) method is called, as n_ref_ports is changed.
        ind = self.proc_params["n_ref_ports"]
        new_ref_port_name = "ref_port_" + str(ind)
        new_mem_var_rp_name = "mem_var_rp_" + str(ind)

        # Create and set new RefPort and corresponding Var to store data
        # Use corresponding container names as parent_list_name.
        setattr(self, new_ref_port_name,
                RefPort(shape=shape,
                        parent_list_name=self.ref_port_cont_nm))
        setattr(self, new_mem_var_rp_name,
                Var(shape=(num_steps,) + shape, init=0,
                    parent_list_name=self.mem_var_for_rp_cont_nm))

        # Add the names of new RefPort and Var name to the corresponding
        # proc_params dict if they are not already part of it
        self.proc_params["ref_ports"].append(new_ref_port_name) if \
            new_ref_port_name not in self.proc_params["ref_ports"] else \
            self.proc_params["ref_ports"]
        self.proc_params["mem_vars_rp"].append(new_mem_var_rp_name) if \
            new_mem_var_rp_name not in self.proc_params["mem_vars_rp"] else \
            self.proc_params["mem_vars_rp"]

        # Register newly created RefPort and Var
        attrs = self._find_attr_by_type(RefPort)
        self._init_proc_member_obj(attrs)
        self.ref_ports.add_members(attrs)

        attrs = self._find_attr_by_type(Var)
        self._init_proc_member_obj(attrs)
        self.vars.add_members(attrs)

    def create_in_port_and_mem_var(self, shape, num_steps):
        """
        Create a new InPort and (memory) Var inside the process. It assigns
        corresponding container names from the Monitor instance.
        Note: The number of InPorts (proc_params["n_in_ports"]) is not
        updated here, but rather inside the probe function, because in some
        cases it might be desirable to create the port and later override them.

        Parameters
        ----------
        shape : tuple
            The shape of InPort will be created. This is used to
            determine the shape of Var required to store probed data

        num_steps : int
            Number of steps the (memory) Var will record the target,
            this is used as last dimension of the shape of Var

        Returns
        -------

        """
        # Create names for InPorts and Vars to be created in Monitor process
        # for OutPort probing purposes. Names are given incrementally each time
        # probe(..) method is called, as n_in_ports is changed.
        ind = self.proc_params["n_in_ports"]
        new_in_port_name = "in_port_" + str(ind)
        new_mem_var_ip_name = "mem_var_ip_" + str(ind)

        # Create and set new InPort and corresponding Var to store data
        # Use corresponding container names as parent_list_name.
        setattr(self, new_in_port_name,
                InPort(shape=shape,
                       parent_list_name=self.in_port_cont_nm))
        setattr(self, new_mem_var_ip_name,
                Var(shape=(num_steps,) + shape, init=0,
                    parent_list_name=self.mem_var_for_ip_cont_nm))

        # Add the names of new InPort and Var name to the corresponding
        # proc_params dict if they are not already part of it
        self.proc_params["in_ports"].append(new_in_port_name) if \
            new_in_port_name not in self.proc_params["in_ports"] else \
            self.proc_params["in_ports"]
        self.proc_params["mem_vars_ip"].append(new_mem_var_ip_name) if \
            new_mem_var_ip_name not in self.proc_params["mem_vars_ip"] else \
            self.proc_params["mem_vars_ip"]

        # Register newly created InPort and Var
        attrs = self._find_attr_by_type(InPort)
        self._init_proc_member_obj(attrs)
        self.in_ports.add_members(attrs)

        attrs = self._find_attr_by_type(Var)
        self._init_proc_member_obj(attrs)
        self.vars.add_members(attrs)

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

        # If target to be monitored is a Var
        if isinstance(target, Var):
            # First create a new RefPort and (memory) Var
            self.create_ref_port_and_mem_var(target.shape, num_steps)

            # Connect newly created RefPort to the Var to be monitored.
            # Access the name of this port using n_ref_ports as index of
            # proc_params["ref_ports"]
            getattr(self, self.proc_params["ref_ports"][
                self.proc_params["n_ref_ports"]]).connect_var(target)

            # Add the name of probed Var and its process to the target_names
            self.target_names[self.proc_params["mem_vars_rp"][
                self.proc_params["n_ref_ports"]]] = [target.process.name,
                                                     target.name]

            # Update n_ref_ports for the next use of probe(..) method
            self.proc_params["n_ref_ports"] += 1

        # If target to be monitored is an OutPort
        elif isinstance(target, OutPort):

            self.create_in_port_and_mem_var(target.shape, num_steps)

            # Connect newly created InPort to the OutPort to be monitored.
            # Access the name of this port using n_in_ports as index of
            # proc_params["in_ports"]
            getattr(self, self.proc_params["in_ports"][
                self.proc_params["n_in_ports"]]).connect_from(target)

            # Add the name of probed OutPort and its process to the target_names
            self.target_names[self.proc_params["mem_vars_ip"][
                self.proc_params["n_in_ports"]]] = [target.process.name,
                                                    target.name]

            # Update n_in_ports for the next use of probe(..) method
            self.proc_params["n_in_ports"] += 1

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
        Fetch the monitoring data from the Vars of Monitor process that
        collected it during the run from probed process, puts into dict form
        for easier access by user.

        Returns
        -------
        data : dict
            Data dictionary collected by Monitor Process
        """

        # Fetch data from memory Vars of Monitor process for monitored Vars
        for i in range(self.proc_params["n_ref_ports"]):
            data_var_name = self.proc_params["mem_vars_rp"][i]
            data_var = getattr(self, data_var_name)
            target_name = self.target_names[data_var_name]

            self.data[target_name[0]][target_name[1]] = data_var.get()

        # Fetch data from memory Vars of Monitor process for monitored OutPorts
        for i in range(self.proc_params["n_in_ports"]):
            data_var_name = self.proc_params["mem_vars_ip"][i]
            data_var = getattr(self, data_var_name)
            target_name = self.target_names[data_var_name]

            self.data[target_name[0]][target_name[1]] = data_var.get()

        return self.data
