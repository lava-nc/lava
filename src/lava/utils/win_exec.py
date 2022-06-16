# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from sys import platform


try:
    from IPython import get_ipython
except ModuleNotFoundError:
    pass

def export(filename, cell):
    '''Exports the cell to the given file name if system is windows.'''
    if platform == "win32" or platform == "cygwin":
        with open(f"{filename}.py", "a") as f:
            f.write(cell)
    else:
        get_ipython().ex(cell)


def load_ipython_extension(shell):
    '''Registers the magic function when the extension loads.'''
    shell.register_magic_function(export, 'cell')


def unload_ipython_extension(shell):
    '''Unregisters the magic function when the extension unloads.'''
    del shell.magics_manager.magics['cell']['export']

if __name__ == "__main__":
    pass
