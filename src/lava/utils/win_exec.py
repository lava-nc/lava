# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from sys import platform
import os


def export_notebook(nb_name: str):
    """Enables execution of Lava tutorials on Windows systems. Processes
    defined within the tutorial jupyter notebook need to be written to a python
    script and imported in the notebook again."""
    if platform == "win32" or platform == "cygwin":
        # Convert the ipython notebook to a python script
        os.system("jupyter nbconvert --to script " + nb_name + ".ipynb")

        # Remove code after definition to avoid execution during import
        with open(nb_name + ".py", "r+") as f:
            d = f.readlines()
            f.seek(0)
            for i in d:
                if i.strip("\n") != "# #### Exception for Windows":
                    f.write(i)
                elif i.strip("\n") == "# #### Exception for Windows":
                    break
            f.truncate()


def cleanup(nb_name: str):
    """Removes previously created python script for tutorial execution on
    Windows systems."""
    if platform == "win32" or platform == "cygwin":
        os.system("del " + nb_name + ".py")
    else:
        os.remove(nb_name + ".py")


if __name__ == "__main__":
    pass
