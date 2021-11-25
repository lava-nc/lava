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


class Loihi1SimCfg(RunConfig):
    """Run configuration selects appropriate ProcessModel -- either
    `SubProcessModel` for a hierarchical Process or else a `PyProcessModel`
    for a standard Process. Implements and overrides `select()` method of the
    parent class `RunConfig` to achieve this.

    First, exceptions are checked in `exception_proc_model_map` dictionary
    as key-value pairs `{Process: ProcessModel}`. These explicit
    specifications are given precedence over the subsequent logic for
    `ProcessModel` selection

    If any `SubProcessModel`s are available, then they will be returned first.

    In case of `PyProcessModel`s, an appropriate `PyProcessModel` is selected
    based on the tags associated with it. Tags are set using the `@tag`
    decorator. If no tags are set for `ProcessModel`s and no tag is selected,
    then the first matching `ProcessModel` will be returned.
    """

    def __init__(self,
                 custom_sync_domains=None,
                 select_tag=None,
                 select_sub_proc_model=False,
                 exception_proc_model_map=None):
        super().__init__(custom_sync_domains=custom_sync_domains)
        self.select_tag = select_tag
        self.select_sub_proc_model = select_sub_proc_model
        self.exception_proc_model_map = exception_proc_model_map
        if not exception_proc_model_map:
            self.exception_proc_model_map = {}

    def select(self, proc, proc_models):
        from lava.magma.core.model.sub.model import AbstractSubProcessModel
        from lava.magma.core.model.py.model import AbstractPyProcessModel
        # First priority to the exceptions. We will simply return the
        # ProcessModel class associated with a Process class in the
        # exceptions dictionary
        if proc.__class__ in self.exception_proc_model_map:
            return self.exception_proc_model_map[proc.__class__]
        # Now, we loop over all ProcessModels proc_models of a Process proc
        for pm in proc_models:
            # Priority given to SubProcessModels
            if issubclass(pm, AbstractSubProcessModel) and \
                    self.select_sub_proc_model:
                return pm
            elif issubclass(pm, AbstractPyProcessModel):
                # If ProcessModel has tags AND user asked for a specific tag
                if len(pm.tags) > 0 and (self.select_tag in pm.tags):
                    return pm
                # If ProcessModel has no tags and no one asked for a tag anyway
                elif len(pm.tags) == 0 and not self.select_tag:
                    return pm
        # We could not find any ProcessModel in the end:
        raise AssertionError(f"No ProcessModel could be selected with "
                             f"tag '{self.select_tag}' for Process "
                             f"{proc.name}::{proc.__class__.__qualname__}.")


class Loihi1HwCfg(RunConfig):
    """A RunConfig for executing model on Loihi 1 HW."""

    pass


class Loihi2SimCfg(RunConfig):
    """A RunConfig for simulating a Loihi 2 model CPU/GPU."""

    pass


class Loihi2HwCfg(RunConfig):
    """A RunConfig for executing model on Loihi 2 HW."""

    pass
