# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from __future__ import annotations
import logging
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
    for a standard Process.

    The following set of rules is applied, in that order of precedence:

    1. A dictionary of exceptions `exception_proc_model_map` is checked first,
    in which user specifies key-value pairs `{Process: ProcessModel}` and the
    `ProcessModel` is returned.

    2. If there is only 1 `ProcessModel` available:
        (a) If the user does not specifically ask for any tags,
            the `ProcessModel` is returned
        (b) If the user asks for a specific tag, then the `ProcessModel` is
            returned only if the tag is found in its list of tags.

    3. If there are multiple `ProcessModel`s available:
        (a) If the user asks specifically to look for `SubProcessModel`s and
            they are available,
            (i)   If there is only 1 `SubProcessModel` available,
                  it is returned
            (ii)  If the user did not ask for any specific tags, the first
                  available `SubProcessModel` is returned
            (iii) If user asked for a specific tag, the first valid
                 `SubProcessModel` is returned, which has the tag in its
                 tag-list
        (b) If user did not explicitly ask for `SubProcessModel`s
            (i)   If the user did not also ask for any specific tag, then the
                  first available `PyProcessModel` is returned
            (ii)  If the user asked for a specific tag, the `PyProcessModel`
                  which has the tag in its tag-list is returned

    Parameters
    ----------
    custom_sync_domains : List[SyncDomain]
                          list of synchronization domains
    select_tag : str
                 ProcessModels with this tag need to be selected
    select_sub_proc_model : bool
                            preferentially select SubProcessModel when True
                            and return
    exception_proc_model_map: (Dict[AbstractProcess, AbstractProcessModel])
        explicit dictionary of {Process: ProcessModel} classes, provided as
        exceptions to the ProcessModel selection logic. The choices made in this
        dict are respected over any logic. For example, {Dense: PyDenseModel}.
        Note that this is a dict mapping classnames to classnames.
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
               proc: AbstractProcess,
               proc_models: ty.List[ty.Type[AbstractProcessModel]]) \
            -> ty.Type[AbstractProcessModel]:
        """
        Selects an appropriate ProcessModel from a list of ProcessModels for
        a Process, based on user requests.

        Parameters
        ----------
        proc: (AbstractProcess) Process for which ProcessModel is selected
        proc_models: (List[AbstractProcessModel]) list of ProcessModels of
            Process

        Returns
        -------
        Selected ProcessModel class
        """

        num_pm = len(proc_models)

        # Case 0: No ProcessModels exist:
        # ------------------------------
        # Raise error
        if num_pm == 0:
            raise AssertionError(f"[{self.__class__.__qualname__}]: No "
                                 f"ProcessModels exist for Process "
                                 f"{proc.name}::{proc.__class__.__qualname__}.")

        # Required modules and helper functions
        from lava.magma.core.model.sub.model import AbstractSubProcessModel
        from lava.magma.core.model.py.model import AbstractPyProcessModel

        def _issubpm(pm: ty.Type[AbstractProcessModel]) -> bool:
            """Checks if input ProcessModel is a SubProcessModel"""
            return issubclass(pm, AbstractSubProcessModel)

        def _ispypm(pm: ty.Type[AbstractProcessModel]) -> bool:
            """Checks if input ProcessModel is a PyProcessModel"""
            return issubclass(pm, AbstractPyProcessModel)

        # Case 1: Exceptions in a dict:
        # ----------------------------
        # We will simply return the ProcessModel class associated with a
        # Process class in the exceptions dictionary
        if proc.__class__ in self.exception_proc_model_map:
            return self.exception_proc_model_map[proc.__class__]

        # Case 2: Only 1 PM found:
        # -----------------------
        # Assumption: User doesn't care about the type: Sub or Py.
        if num_pm == 1:
            # If type of the PM is neither Sub nor Py, raise error
            if not (_issubpm(proc_models[0]) or _ispypm(proc_models[0])):
                raise NotImplementedError(f"[{self.__class__.__qualname__}]: "
                                          f"The only found ProcessModel "
                                          f"{proc_models[0].__qualname__} is "
                                          f"neither a SubProcessModel nor a "
                                          f"PyProcessModel. Not supported by "
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
                    # ToDo: Currently we are silently returning an untagged PM
                    #  here. This might become a root-cause for errors in the
                    #  future.
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
                            f"{proc.name}::"
                            f"{proc.__class__.__qualname__}.")

        # Case 3: Multiple PMs exist:
        # --------------------------
        # Collect indices of Sub and Py PMs:
        sub_pm_idxs = [idx for idx, pm in enumerate(proc_models) if
                       _issubpm(pm)]
        py_pm_idxs = [idx for idx, pm in enumerate(proc_models) if _ispypm(pm)]
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
                self.log.info(f"[{self.__class__.__qualname__}]: Using the"
                              f" first SubProcessModel "
                              f"{proc_models[sub_pm_idxs[0]].__qualname__} "
                              f"available for Process "
                              f"{proc.name}::{proc.__class__.__qualname__}.")
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
                                         f"{proc.name}::"
                                         f"{proc.__class__.__qualname__}.")
                # ToDo: Currently we check for only 1 tag. So we return the
                #  first SubPM with select_tag.
                return proc_models[valid_sub_pm_idxs[0]]
        # Case 3b: User didn't ask for SubProcessModel:
        # --------------------------------------------
        # Raise error if no PyProcessModels exist
        if len(py_pm_idxs) == 0:
            raise AssertionError(f"[{self.__class__.__qualname__}]: "
                                 f"No PyProcessModels were "
                                 f"found for Process {proc.name}::"
                                 f"{proc.__class__.__qualname__}. "
                                 f"Try setting select_sub_proc_model=True.")
        # Case 3b(i): User didn't provide select_tag:
        # ------------------------------------------
        # Assumption: User doesn't care about tags. We return the first
        # PyProcessModel found
        if self.select_tag is None:
            self.log.info(f"[{self.__class__.__qualname__}]: Using the first "
                          f"PyProcessModel "
                          f"{proc_models[py_pm_idxs[0]].__qualname__} "
                          f"available for Process "
                          f"{proc.name}::{proc.__class__.__qualname__}.")
            return proc_models[py_pm_idxs[0]]
        # Case 3b(ii): User asked for a specific tag:
        # ------------------------------------------
        else:
            # Collect indices of all PyPMs with select_tag
            valid_py_pm_idxs = \
                [idx for idx in py_pm_idxs
                 if self.select_tag in proc_models[py_pm_idxs[idx]].tags]
            if len(valid_py_pm_idxs) == 0:
                raise AssertionError(f"[{self.__class__.__qualname__}]: No "
                                     f"ProcessModels found with tag "
                                     f"'{self.select_tag}' for Process "
                                     f"{proc.name}::"
                                     f"{proc.__class__.__qualname__}.")
            # ToDo: Currently we check for only 1 tag. So we return the
            #  first PyPM with select_tag.
            return proc_models[valid_py_pm_idxs[0]]


class Loihi1HwCfg(RunConfig):
    """A RunConfig for executing model on Loihi 1 HW."""

    pass


class Loihi2SimCfg(RunConfig):
    """A RunConfig for simulating a Loihi 2 model CPU/GPU."""

    pass


class Loihi2HwCfg(RunConfig):
    """A RunConfig for executing model on Loihi 2 HW."""

    pass
