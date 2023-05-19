# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod

from lava.magma.core.model.model import AbstractProcessModel


class AbstractBuilder(ABC):
    """Builders interface for building processes in a given backend."""

    @abstractmethod
    def build(self):
        """Build the actual process."""


class ResourceAddress(ABC):
    pass


class Resource(ABC):
    def write(self, hw: ty.Any):
        """Given hw, write this compiled resource"""


class CompiledResource(Resource):
    """Signifies a compiled resource held by the builder. Must be
    serializable if the builder is being serialized after compilation
    before mapping"""

    @property
    def l_address(self) -> ResourceAddress:
        """Return the logical address of this compiled resource."""
        raise NotImplementedError


class MappedResource(Resource):
    """Signifies a physical resource held by the builder.
    Must be serializable."""

    @property
    @abstractmethod
    def p_address(self) -> ResourceAddress:
        """Return the physical address of this mapped resource."""
        pass


class AbstractProcessBuilder(AbstractBuilder):
    """An AbstractProcessBuilder is the base type for process builders.

    Process builders instantiate and initialize a ProcessModel.

    Parameters
    ----------

    proc_model: AbstractProcessModel
                ProcessModel class of the process to build.
    model_id: int
              model_id represents the ProcessModel ID to build.

    """

    def __init__(
            self,
            proc_model: ty.Type[AbstractProcessModel],
            model_id: int):
        self.var_id_to_var_model_map: \
            ty.Dict[int,
                    ty.Type["AbstractVarModel"]] = {}
        self._proc_model = proc_model
        self._model_id = model_id

    @property
    def proc_model(self) -> ty.Type[AbstractProcessModel]:
        return self._proc_model

    def _check_members_exist(self, members: ty.Iterable, m_type: str):
        """Checks that ProcessModel has same members as Process.

        Parameters
        ----------
        members : ty.Iterable

        m_type : str

        Raises
        ------
        AssertionError
            Process and ProcessModel name should match
        """
        proc_name = self.proc_model.implements_process.__name__
        proc_model_name = self.proc_model.__name__
        for m in members:
            if not hasattr(self.proc_model, m.name):
                raise AssertionError(
                    "Both Process '{}' and ProcessModel '{}' are expected to "
                    "have {} named '{}'.".format(
                        proc_name, proc_model_name, m_type, m.name
                    )
                )

    @staticmethod
    def _check_not_assigned_yet(
            collection: dict, keys: ty.Iterable[str], m_type: str
    ):
        """Checks that collection dictionary not already contain given keys
        to prevent overwriting of existing elements.

        Parameters
        ----------
        collection : dict

        keys : ty.Iterable[str]

        m_type : str


        Raises
        ------
        AssertionError

        """
        for key in keys:
            if key in collection:
                raise AssertionError(
                    f"Member '{key}' already found in {m_type}."
                )

    def set_variables(self, variables: ty.List["VarInitializer"]):
        """Appends the given list of variables to the ProcessModel. Used by the
         compiler to create a ProcessBuilder during the compilation of
         ProcessModels.

        Parameters
        ----------
        variables : ty.List[VarInitializer]

        """
        self._check_members_exist(variables, "Var")
        new_vars = {v.name: v for v in variables}
        self._check_not_assigned_yet(self.vars, new_vars.keys(), "vars")
        self.vars.update(new_vars)


class AbstractChannelBuilder(ABC):
    """An AbstractChannelBuilder is the base type for
    channel builders which build communication channels
    between services and processes"""

    pass
