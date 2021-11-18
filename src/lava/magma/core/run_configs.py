# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from __future__ import annotations
import typing as ty
from abc import ABC

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.resources import AbstractNode
from lava.magma.core.sync.domain import SyncDomain


# ToDo: Draft interface and provide realizations for pure Python
#  execution and for execution on Loihi corresponding to our former
#  N2-HW, N3-HW, N2-Sim and N3-Sim backends.
# ToDo: Provide an example how pre-configured RunCfg can be customized.
class RunConfig(ABC):
    """Could just have a static 'select' method that contains arbitrary code
    how to select a ProcessModel of a Process to get started instead of
    formal rules.

    E.g. If there exists a Loihi1 ProcessModel, pick that. Else if there is
    an EMB ProcessModel, pick that. Or Just always pick PyProcessModel.

    This method could also filter for other tags of  ProcessModel such as
    bit-accurate, floting point, etc.

    RunConfig can also specify how many nodes of a certain type will be
    available. This will allows to allocate all RuntimeService processes
    during compilation process. Downside of this is tht we have to decide
    very early about the cluster size or Loihi system type although this
    could in principle be decided in Runtime. But if we only were to decide
    this in Runtime, then we would still have to create specific instances of
    RuntimeService processes and corresponding channel configuration in
    Runtime which is actually a compiler job.
    As a mitigation one could clearly break compiler and Executable
    generation up into stages that can be called separately. Then once could
    still change such configuration details later and repeat the required
    compilation stages or just do them in the first place.
    """

    def __init__(self,
                 custom_sync_domains: ty.Optional[ty.List[SyncDomain]] = None):
        self.custom_sync_domains = []
        if custom_sync_domains:
            if not isinstance(custom_sync_domains, list):
                raise AssertionError("'custom_sync_domains' must be a list.")
            for sd in custom_sync_domains:
                if not isinstance(sd, SyncDomain):
                    raise AssertionError("Expected list of SyncDomains.")
            self.custom_sync_domains += custom_sync_domains

    def exclude_nodes(self, nodes: ty.List[AbstractNode]):
        """Excludes given nodes from consideration by compiler."""
        pass

    def require_nodes(self, nodes: ty.List[AbstractNode]):
        """Requires that compiler maps processes to given nodes."""
        pass

    def select(self,
               process: AbstractProcess,
               proc_model: ty.List[ty.Type[AbstractProcessModel]]) \
            -> ty.Type[AbstractProcessModel]:
        pass


# ToDo: This is only a minimal RunConfig that will select SubProcModel if there
#       is one and if not selects a PyProcessModel with the correct tag.
#       Needs to be modified in future releases to support more complicated
#       sync domains, @requires (example: GPU support), and to select
#       LeafProcModels of type other than PyProcessModel.
class Loihi1SimCfg(RunConfig):
    """Run configuration selects appropriate ProcessModel -- either SubProcessModel for a
     Hierarchical Process or else a PyProcessModel for a standard Process. The appropriate
     PyProcessModel is selected based @tag('floating_pt') or @tag('fixed_pt'), for
    floating point precision or Loihi bit-accurate fixed point precision respectively"""

    def __init__(self, custom_sync_domains=None, select_tag='floating_pt', select_sub_proc_model=False):
        super().__init__(custom_sync_domains=custom_sync_domains)
        self.select_tag = select_tag
        self.select_sub_proc_model = select_sub_proc_model

    def select(self, proc, proc_models):
        from lava.magma.core.model.sub.model import AbstractSubProcessModel
        from lava.magma.core.model.py.model import AbstractPyProcessModel
        py_proc_model = None
        sub_proc_model = None
        for pm in proc_models:
            if issubclass(pm, AbstractSubProcessModel):
                sub_proc_model = pm
            if issubclass(pm, AbstractPyProcessModel):
                py_proc_model = pm
                # Make selection
            if self.select_sub_proc_model and sub_proc_model:
                return sub_proc_model
            elif py_proc_model:
                if self.select_tag in py_proc_model.tags:
                    return py_proc_model
            else:
                raise AssertionError("No legal ProcessModel found.")


class Loihi1HwCfg(RunConfig):
    """A RunConfig for executing model on Loihi 1 HW."""

    pass


class Loihi2SimCfg(RunConfig):
    """A RunConfig for simulating a Loihi 2 model CPU/GPU."""

    pass


class Loihi2HwCfg(RunConfig):
    """A RunConfig for executing model on Loihi 2 HW."""

    pass
