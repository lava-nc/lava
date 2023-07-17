# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import pickle  # noqa: S403 # nosec
import typing as ty
import os

from lava.magma.core.process.process import AbstractProcess
from lava.magma.compiler.executable import Executable


class SerializationObject:
    """This class is used to serialize a process or a list of processes
    together with a corresponding executable.

    Parameters
    ----------
    processes: AbstractProcess, ty.List[AbstractProcess]
        A process or a list of processes which should be stored in a file.
    executable: Executable, optional
        The corresponding executable of the compiled processes which should be
        stored in a file.
    """
    def __init__(self,
                 processes: ty.Union[AbstractProcess,
                                     ty.List[AbstractProcess]],
                 executable: ty.Optional[Executable] = None) -> None:

        self.processes = processes
        self.executable = executable


def save(processes: ty.Union[AbstractProcess, ty.List[AbstractProcess]],
         filename: str,
         executable: ty.Optional[Executable] = None) -> None:
    """Saves a given process or list of processes with an (optional)
    corresponding executable to file <filename>.

    Parameters
    ----------
    processes: AbstractProcess, ty.List[AbstractProcess]
        A process or a list of processes which should be stored in a file.
    filename: str
        The path + name of the file. If no file extension is given,
        '.pickle' will be added automatically.
    executable: Executable, optional
        The corresponding executable of the compiled processes which should be
        stored in a file.

    Raises
    ------
    TypeError
        If argument <process> is not AbstractProcess, argument <filename> is
        not string or argument <executable> is not Executable.
    """
    # Check parameter types
    if not isinstance(processes, list) and not isinstance(processes,
                                                          AbstractProcess):
        raise TypeError(f"Argument <processes> must be AbstractProcess"
                        f" or list of AbstractProcess, but got"
                        f" {processes}.")
    if not isinstance(filename, str):
        raise TypeError(f"Argument <filename> must be string"
                        f" but got {filename}.")
    if executable is not None and not isinstance(executable, Executable):
        raise TypeError(f"Argument <executable> must be Executable"
                        f" but got {executable}.")

    # Create object which is stored
    obj = SerializationObject(processes, executable)

    # Add default file extension if no extension is present
    if "." not in filename:
        filename = filename + ".pickle"

    # Store object at <filename>
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load(filename: str) -> ty.Tuple[ty.Union[AbstractProcess,
                                             ty.List[AbstractProcess]],
                                    ty.Union[None, Executable]]:
    """Loads a process or list of processes with an (optional)
    corresponding executable from file <filename>.

    Parameters
    ----------
    filename: str
        The path + name of the file. If no file extension is given,
        '.pickle' will be added automatically.

    Returns
    -------
    tuple
        Returns a tuple of a process or list of processes and a executable or
        None.

    Raises
    ------
    OSError
        If the input file does not exist or cannot be read.
    TypeError
        If argument <filename> is not a string.
    AssertionError
        If provided file is not compatible/contains unexpected data.
    """

    # Check parameter types
    if not isinstance(filename, str):
        raise TypeError(f"Argument <filename> must be string"
                        f" but got {filename}.")

    # Check if filename exists
    if not os.path.isfile(filename):
        raise OSError(f"File {filename} could not be found.")

    # Load serialized object from <filename>
    with open(filename, 'rb') as f:
        obj = pickle.load(f)  # noqa: S301 # nosec

    # Check loaded object
    if not isinstance(obj, SerializationObject):
        raise AssertionError(f"Incompatible file {filename} was provided.")

    # Return processes and executable
    return obj.processes, obj.executable
