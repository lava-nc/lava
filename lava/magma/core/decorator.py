# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import AbstractResource
from lava.magma.core.sync.protocol import AbstractSyncProtocol


def has_models(*args: ty.Type[AbstractProcessModel]):
    """Decorates Process class by adding all ProcessModel classes for it
    'has_models' will fail if an attempt is made to overwrite an already set
    Process class of a ProcessModel class.

    Example: @has_models(Model1, Model2, Model3)
    """
    proc_models = list(args)

    for proc_model in proc_models:
        if not issubclass(proc_model, AbstractProcessModel):
            raise AssertionError("'has_models' accepts individual "
                                 "'AbstractProcessModel'.")

    def decorate_process(cls: type):
        if not issubclass(cls, AbstractProcess):
            raise AssertionError("Decorated class must be a subclass "
                                 "of 'AbstractProcess'.")

        for process_model in proc_models:
            if process_model.implements_process and \
                    process_model.implements_process != cls:
                raise AssertionError(
                    f"ProcessModel '{process_model.__name__}' already"
                    f" implements a Process.")

        setattr(cls, "process_models", proc_models)
        for process_model in proc_models:
            setattr(process_model, "implements_process", cls)

        return cls

    return decorate_process


def implements_protocol(protocol: ty.Type[AbstractSyncProtocol]):
    """Decorates ProcessModel class by adding the SyncProtocol that
        this ProcessModel implements as a class variable.

    'implements' will fail if an attempt is made to overwrite
        an already set SyncProtocol class of a parent class.

    Parameters
    ----------
    protocol: The SyncProtocol tht the ProcessModel implements.
    """

    if not issubclass(protocol, AbstractSyncProtocol):
        raise AssertionError

    def decorate_process_model(cls: type):
        if not issubclass(cls, AbstractProcessModel):
            raise AssertionError("Decorated class must be a \
            subclass of 'AbstractProcessModel'.")

        # Check existing 'implements_protocol' does not get overwritten by a
        # different protocol
        if cls.implements_protocol and \
                cls.implements_protocol != protocol:
            raise AssertionError(
                f"ProcessModel '{cls.__name__}' already implements a "
                f"SyncProtocol (perhaps due to sub classing).")

        # Reset attribute on this class to not overwrite parent class
        setattr(cls, 'implements_protocol', protocol)

        return cls

    return decorate_process_model


def requires(*args: ty.Union[ty.Type[AbstractResource],
                             ty.List[ty.Type[AbstractResource]]]):
    """Decorator for ProcessModel classes that adds class variable to
    ProcessModel class that specifies which resources the ProcessModel
    requires.
    In order to express optionality between one or more resources, include
    them in a list or tuple.

    Example: @requires(Res1, Res2, [Res3, Res4])
        -> Requires Res1 and Res2 and one of Res3 or Res4
    """

    reqs = list(args)

    for req in reqs:
        if not isinstance(req, list) and not issubclass(req, AbstractResource):
            raise AssertionError("'requires' accepts individual or "
                                 "lists of 'AbstractResources'.")
        if isinstance(req, list):
            for r in req:
                if not issubclass(r, AbstractResource):
                    raise AssertionError("Lists passed to 'require' must \
                    contain subclasses of AbstractResource.")

    def decorate_process_model(cls: type):
        if not issubclass(cls, AbstractProcessModel):
            raise AssertionError("Decorated class must be a subclass "
                                 "of 'AbstractProcessModel'.")

        # Get requirements of parent class
        super_res = cls.required_resources.copy()
        # Set new requirements on this cls to not overwrite parent class
        # requirements
        setattr(cls, 'required_resources', super_res + reqs)
        return cls

    return decorate_process_model
