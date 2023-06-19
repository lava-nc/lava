# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from __future__ import annotations
import logging
import typing as ty
from abc import ABC
from itertools import chain
from lava.magma.core.resources import AbstractNode, Loihi1NeuroCore, \
    Loihi2NeuroCore, NeuroCore
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.callback_fx import CallbackFx

try:
    from lava.magma.core.model.c.model import CLoihiProcessModel
    from lava.magma.core.model.nc.model import AbstractNcProcessModel
except ImportError:
    class AbstractCProcessModel:
        pass

    class AbstractNcProcessModel:
        pass

from lava.magma.core.sync.domain import SyncDomain
from lava.magma.compiler.subcompilers.constants import \
    EMBEDDED_ALLOCATION_ORDER

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.core.model.model import AbstractProcessModel


class RunConfig(ABC):
    """Basic run configuration and base class for other run configurations.

    A RunConfig specifies how to execute Processes on a specific hardware
    backend. Its main purpose is to select the appropriate ProcessModels
    given the Processes to be executed and the given tags (i.e. bit-accurate,
    floating, etc) using the `select()` function.


    A RunConfig allows the user to guide the compiler in its choice of
    ProcessModels. When the user compiles/runs a Process for the first time,
    a specific RunConfig must be provided. The compiler will
    follow the selection rules laid out in the select() method of the
    RunConfig to choose the optimal ProcessModel for the Process.

    A RunConfig can filter the ProcessModels by various criteria.
    Examples include the preferred computing resource or user-defined tags.
    It may also specify how many computing nodes of a certain type,
    like embedded CPUs, will be available. This will allow to allocate all
    RuntimeService processes during compilation. A RunConfig can also give hints
    to the compiler which computational nodes are required and which are
    excluded.

    Parameters
    ----------
    custom_sync_domains : List[SyncDomain]
        List of user-specified synchronization domains.
    loglevel: int
              Sets level of event logging, as defined by Python's 'logging'
              facility. Default: logging.WARNING
    """

    def __init__(self,
                 custom_sync_domains: ty.Optional[ty.List[SyncDomain]] = None,
                 loglevel: int = logging.WARNING):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(loglevel)
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

    def require_nodes(self, nodes: ty.List[AbstractNode]):
        """Requires that compiler maps processes to given nodes."""

    def select(self,
               process: AbstractProcess,
               proc_model: ty.List[ty.Type[AbstractProcessModel]]) \
            -> ty.Type[AbstractProcessModel]:
        pass


class AbstractLoihiRunCfg(RunConfig):
    """Selects the appropriate ProcessModel for Loihi RunConfigs.

    The following set of rules is applied, in that order of precedence:

    1. A dictionary of exceptions `exception_proc_model_map` is checked first,
    in which user specifies key-value pairs `{Process: ProcessModel}` and the
    `ProcessModel` is returned.

    2. If there is only 1 `ProcessModel` available:

        - If the user does not specifically ask for any tags,
          the `ProcessModel` is returned
        - If the user asks for a specific tag, then the `ProcessModel` is
          returned only if the tag is found in its list of tags.

    3. If there are multiple `ProcessModel`s available:

        - If the user asks specifically to look for `SubProcessModel`s and
          they are available:

             - If there is only 1 `SubProcessModel` available, it is returned
             - If the user did not ask for any specific tags, the first
               available `SubProcessModel` is returned
             - If user asked for a specific tag, the first valid
               `SubProcessModel` is returned, which has the tag in its tag-list

        - If user did not explicitly ask for `SubProcessModel`s:

            - If the user did not also ask for any specific tag, then the
              first available ProcessModel is returned that requires the
              correct computing hardware.
            - If the user asked for a specific tag,
              the hardware-specific ProcessModel which has the tag in its
              tag-list is returned

    Parameters
    ----------
    custom_sync_domains : List[SyncDomain]
        list of synchronization domains
    select_tag : str
        The RunConfig will select only ProcessModels that have the tag
        'select_tag'.
        Example: By setting select_tag="fixed_pt", it will select ProcessModels
        that implement a fixed-point implementation of the Lava Processes in
        the architecture that is to be executed.
    select_sub_proc_model : bool
        When set to True, hierarchical SubProcessModels are selected over
        LeafProcessModels, where available.
    exception_proc_model_map: (Dict[AbstractProcess, AbstractProcessModel])
        explicit dictionary of {Process: ProcessModel} classes, provided as
        exceptions to the ProcessModel selection logic. The choices made in this
        dict are respected over any logic. For example, {Dense: PyDenseModel}.
        Note that this is a dict mapping classnames to classnames.
    loglevel: int
        sets level of event logging, as defined by Python's 'logging'
        facility. Default: logging.WARNING
    """

    def __init__(self,
                 custom_sync_domains: ty.Optional[ty.List[SyncDomain]] = None,
                 select_tag: ty.Optional[str] = None,
                 select_sub_proc_model: ty.Optional[bool] = False,
                 exception_proc_model_map: ty.Optional[ty.Dict[
                     ty.Type[AbstractProcess], ty.Type[
                         AbstractProcessModel]]] = None,
                 loglevel: int = logging.WARNING):
        super().__init__(custom_sync_domains=custom_sync_domains,
                         loglevel=loglevel)
        self.select_tag = select_tag
        self.select_sub_proc_model = select_sub_proc_model
        self.exception_proc_model_map = exception_proc_model_map
        if not exception_proc_model_map:
            self.exception_proc_model_map = {}

    def select(self,
               process: AbstractProcess,
               proc_models: ty.List[ty.Type[AbstractProcessModel]]) \
            -> ty.Type[AbstractProcessModel]:
        """
        Selects an appropriate ProcessModel from a list of ProcessModels for
        a Process, based on user requests.

        Parameters
        ----------
        process: AbstractProcess
            Process for which ProcessModel is selected
        proc_models: List[AbstractProcessModel]
            List of ProcessModels to select from

        Returns
        -------
        Selected ProcessModel class
        """
        num_pm = len(proc_models)
        # Case 0: No ProcessModels exist:
        # ------------------------------
        # Raise error
        if num_pm == 0:
            raise AssertionError(
                f"[{self.__class__.__qualname__}]: No ProcessModels exist for "
                f"Process {process.name}::{process.__class__.__qualname__}."
            )

        # Required modules and helper functions
        from lava.magma.core.model.sub.model import AbstractSubProcessModel

        def _issubpm(pm: ty.Type[AbstractProcessModel]) -> bool:
            """Checks if input ProcessModel is a SubProcessModel"""
            return issubclass(pm, AbstractSubProcessModel)

        # Case 1: Exceptions in a dict:
        # ----------------------------
        # We will simply return the ProcessModel class associated with a
        # Process class in the exceptions dictionary
        if process.__class__ in self.exception_proc_model_map:
            return self.exception_proc_model_map[process.__class__]

        # Case 2: Only 1 PM found:
        # -----------------------
        # Assumption: User doesn't care about the type: Sub or HW-specific.
        if num_pm == 1:
            # If type of the PM is neither Sub nor HW-supported, raise error
            if not (_issubpm(proc_models[0]) or self._is_hw_supported(
                    proc_models[0])):
                raise NotImplementedError(f"[{self.__class__.__qualname__}]: "
                                          f"The only found ProcessModel "
                                          f"{proc_models[0].__qualname__} is "
                                          f"neither a SubProcessModel nor "
                                          f"runs on a backend supported by "
                                          f"this RunConfig.")
            # Case 2a: User did not provide select_tag:
            # ----------------------------------------
            # Assumption: User doesn't care about tags, that's why none was
            # provided. Just return the only ProcessModel available
            if self.select_tag is None:
                return proc_models[0]
            # Case 2b: select_tag is provided
            else:
                # Case 2b(i) PM is untagged:
                # -------------------------
                # Assumption: User found it unnecessary to tag the PM for this
                # particular process.
                if len(proc_models[0].tags) == 0:
                    return proc_models[0]
                # Case 2b(ii): PM is tagged:
                # -------------------------
                else:
                    if self.select_tag in proc_models[0].tags:
                        return proc_models[0]
                    else:
                        # We did not find the tag that user provided in tags
                        raise AssertionError(
                            f"[{self.__class__.__qualname__}]: No "
                            f"ProcessModels found with tag "
                            f"'{self.select_tag}' for Process "
                            f"{process.name}::"
                            f"{process.__class__.__qualname__}.")

        # Case 3: Multiple PMs exist:
        # --------------------------
        # Collect indices of Sub and HW-specific PMs:
        sub_pm_idxs = [idx for idx, pm in enumerate(proc_models) if
                       _issubpm(pm)]
        leaf_pm_idxs = self._order_according_to_resources(proc_models)
        # Case 3a: User specifically asked for a SubProcessModel:
        # ------------------------------------------------------
        if self.select_sub_proc_model and len(sub_pm_idxs) > 0:
            # Case 3a(i): There is only 1 Sub PM:
            # ----------------------------------
            # Assumption: User wants to use the only SubPM available
            if len(sub_pm_idxs) == 1:
                return proc_models[sub_pm_idxs[0]]
            # Case 3a(ii): User didn't provide select_tag:
            # -------------------------------------------
            # Assumption: User doesn't care about tags. We return the first
            # SubProcessModel found
            if self.select_tag is None:
                self.log.info(
                    f"[{self.__class__.__qualname__}]: Using the first "
                    f"SubProcessModel "
                    f"{proc_models[sub_pm_idxs[0]].__qualname__} "
                    f"available for Process "
                    f"{process.name}::{process.__class__.__qualname__}."
                )
                return proc_models[sub_pm_idxs[0]]
            # Case 3a(iii): User asked for a specific tag:
            # -------------------------------------------
            else:
                # Collect indices of all SubPMs with select_tag
                valid_sub_pm_idxs = \
                    [idx for idx in sub_pm_idxs
                     if self.select_tag in proc_models[sub_pm_idxs[idx]].tags]
                if len(valid_sub_pm_idxs) == 0:
                    raise AssertionError(f"[{self.__class__.__qualname__}]: No "
                                         f"ProcessModels found with tag "
                                         f"{self.select_tag} for Process "
                                         f"{process.name}::"
                                         f"{process.__class__.__qualname__}.")
                return proc_models[valid_sub_pm_idxs[0]]
        # Case 3b: User didn't ask for SubProcessModel:
        # --------------------------------------------
        # Raise error if no HW-specific ProcessModels exist
        if len(leaf_pm_idxs) == 0:
            raise AssertionError(f"[{self.__class__.__qualname__}]: "
                                 f"No hardware-specific ProcessModels were "
                                 f"found for Process {process.name}::"
                                 f"{process.__class__.__qualname__}. "
                                 f"Try setting select_sub_proc_model=True.")
        # Case 3b(i): User didn't provide select_tag:
        # ------------------------------------------
        # Assumption: User doesn't care about tags. We return the first
        # HW-specific ProcessModel found
        if self.select_tag is None:
            self.log.info(f"[{self.__class__.__qualname__}]: Using the first "
                          f"Hardware-specific ProcessModel "
                          f"{proc_models[leaf_pm_idxs[0]].__qualname__} "
                          f"available for Process "
                          f"{process.name}::{process.__class__.__qualname__}.")
            return proc_models[leaf_pm_idxs[0]]
        # Case 3b(ii): User asked for a specific tag:
        # ------------------------------------------
        else:
            # Collect indices of all HW-specific PMs with select_tag
            valid_leaf_pm_idxs = \
                [idx for idx in leaf_pm_idxs
                 if self.select_tag in proc_models[idx].tags]
            if len(valid_leaf_pm_idxs) == 0:
                raise AssertionError(f"[{self.__class__.__qualname__}]: No "
                                     f"ProcessModels found with tag "
                                     f"'{self.select_tag}' for Process "
                                     f"{process.name}::"
                                     f"{process.__class__.__qualname__}.")
            return proc_models[valid_leaf_pm_idxs[0]]

    def _is_hw_supported(self, pm: ty.Type[AbstractProcessModel]) -> bool:
        """Checks if the process models is a PyProcModel"""
        return issubclass(pm, AbstractPyProcessModel)

    def _order_according_to_resources(self, proc_models: ty.List[ty.Type[
            AbstractProcessModel]]) -> ty.List[int]:
        """Orders a list of ProcModels according to the resources that it
        runs on. ProcModels that require unsupported HW are left out. The
        return value is a list of the indices specifying the preferred order.
        This method is should be implemented by the inheriting RunConfig."""
        return list(range(len(proc_models)))


class AbstractLoihiHWRunCfg(AbstractLoihiRunCfg):
    pass


class AbstractLoihiSimRunCfg(AbstractLoihiRunCfg):
    pass


class Loihi1SimCfg(AbstractLoihiSimRunCfg):
    """Run configuration selects appropriate ProcessModel -- either
    `SubProcessModel` for a hierarchical Process or else a `PyProcessModel`
    for a standard Process.
    """

    def _order_according_to_resources(self, proc_models: ty.List[ty.Type[
            AbstractProcessModel]]) -> ty.List[int]:
        """For Sim configurations, only PyProcModels are allowed."""

        proc_models_ordered = [idx for idx, pm in enumerate(proc_models)
                               if issubclass(pm, AbstractPyProcessModel)]
        return proc_models_ordered


class Loihi1HwCfg(AbstractLoihiHWRunCfg):
    """
    A RunConfig for executing model on Loihi1 HW.
    For Loihi1 HW configurations, the preferred ProcModels are NcProcModels
    that can run on a NeuroCore of a Loihi1NeuroCore
    or, if none is found, CProcModels. This preference can be overwritten by
    a tag provided by the user. This RunConfig will default to a PyProcModel
    if no Loihi1-compatible ProcModel is being found.
    ."""

    def __init__(self,
                 custom_sync_domains: ty.Optional[ty.List[SyncDomain]] = None,
                 select_tag: ty.Optional[str] = None,
                 select_sub_proc_model: ty.Optional[bool] = False,
                 exception_proc_model_map: ty.Optional[ty.Dict[
                     ty.Type[AbstractProcess], ty.Type[
                         AbstractProcessModel]]] = None,
                 loglevel: int = logging.WARNING,
                 callback_fxs: ty.List[CallbackFx] = None,
                 embedded_allocation_order=EMBEDDED_ALLOCATION_ORDER.NORMAL):
        super().__init__(custom_sync_domains,
                         select_tag,
                         select_sub_proc_model,
                         exception_proc_model_map,
                         loglevel)
        self.callback_fxs: ty.List[CallbackFx] = [] if not callback_fxs else \
            callback_fxs
        self.embedded_allocation_order: EMBEDDED_ALLOCATION_ORDER = \
            embedded_allocation_order

    def _order_according_to_resources(self, proc_models: ty.List[ty.Type[
            AbstractProcessModel]]) -> ty.List[int]:
        """Orders the provided ProcModels according to the preferences for
        Loihi 1 HW."""
        # PyProcModels
        proc_models_py = [idx for idx, pm in enumerate(proc_models)
                          if issubclass(pm, AbstractPyProcessModel)]
        # NcProcModels compatible with Loihi1 HW
        proc_models_nc = [idx for idx, pm in enumerate(proc_models)
                          if (NeuroCore in pm.required_resources
                              or Loihi1NeuroCore in pm.required_resources)
                          and issubclass(pm, AbstractNcProcessModel)]
        # CProcModels compatible with Loihi
        proc_models_c = [idx for idx, pm in enumerate(proc_models)
                         if issubclass(pm, CLoihiProcessModel)]
        return list(chain(proc_models_nc, proc_models_c, proc_models_py))

    def _is_hw_supported(self, pm: ty.Type[AbstractProcessModel]) -> bool:
        """Checks if the process models is a supporte by Loihi 1 HW."""
        return issubclass(pm, AbstractPyProcessModel) \
            or issubclass(pm, CLoihiProcessModel) \
            or ((NeuroCore in pm.required_resources or Loihi1NeuroCore in
                 pm.required_resources)
                and issubclass(pm, AbstractNcProcessModel))


class Loihi2SimCfg(Loihi1SimCfg):
    """A RunConfig for simulating a Loihi 2 model CPU/GPU."""

    pass


class Loihi2HwCfg(AbstractLoihiHWRunCfg):
    """
    A RunConfig for executing model on Loihi2 HW.
    For Loihi2 HW configurations, the preferred ProcModels are NcProcModels
    that can run on a NeuroCore of a Loihi2NeuroCore
    or, if none is found, CProcModels. This preference can be overwritten by
    a tag provided by the user. This RunConfig will default to a PyProcModel
    if no Loihi2-compatible ProcModel is being found.
    """

    def __init__(self,
                 custom_sync_domains: ty.Optional[ty.List[SyncDomain]] = None,
                 select_tag: ty.Optional[str] = None,
                 select_sub_proc_model: ty.Optional[bool] = False,
                 exception_proc_model_map: ty.Optional[ty.Dict[
                     ty.Type[AbstractProcess], ty.Type[
                         AbstractProcessModel]]] = None,
                 loglevel: int = logging.WARNING,
                 callback_fxs: ty.List[CallbackFx] = None,
                 embedded_allocation_order=EMBEDDED_ALLOCATION_ORDER.NORMAL):
        super().__init__(custom_sync_domains,
                         select_tag,
                         select_sub_proc_model,
                         exception_proc_model_map,
                         loglevel)
        self.callback_fxs: ty.List[CallbackFx] = [] if not callback_fxs else \
            callback_fxs
        self.embedded_allocation_order: EMBEDDED_ALLOCATION_ORDER = \
            embedded_allocation_order

    def _order_according_to_resources(self, proc_models: ty.List[ty.Type[
            AbstractProcessModel]]) -> ty.List[int]:
        """Orders the provided ProcModels according to the preferences for
        Loihi 1 HW."""
        proc_models_py = [idx for idx, pm in enumerate(proc_models)
                          if issubclass(pm, AbstractPyProcessModel)]
        # NcProcModels compatible with Loihi2 HW
        proc_models_nc = [idx for idx, pm in enumerate(proc_models)
                          if (NeuroCore in pm.required_resources
                              or Loihi2NeuroCore in pm.required_resources)
                          and issubclass(pm, AbstractNcProcessModel)]
        # CProcModels compatible with Loihi
        proc_models_c = [idx for idx, pm in enumerate(proc_models)
                         if issubclass(pm, CLoihiProcessModel)]
        # PyProcModels in Loihi2HwCfg will be made available in the future
        return list(chain(proc_models_nc, proc_models_c, proc_models_py))

    def _is_hw_supported(self, pm: ty.Type[AbstractProcessModel]) -> bool:
        """Checks if the process models is a supporte by Loihi 2 HW."""
        return issubclass(pm, AbstractPyProcessModel) \
            or issubclass(pm, CLoihiProcessModel) \
            or ((NeuroCore in pm.required_resources or Loihi2NeuroCore in
                 pm.required_resources)
                and issubclass(pm, AbstractNcProcessModel))
