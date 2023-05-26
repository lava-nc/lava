# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import subprocess  # noqa S404
import importlib


class Utils:
    """Utility Class containing testing helper code that can be reused
    between tests.
    """

    @staticmethod
    def get_bool_env_setting(env_var: str):
        """Get an environment varible and return
        True if the variable is set to 1 else return
        false
        """
        env_test_setting = os.environ.get(env_var)
        test_setting = False
        if env_test_setting == "1":
            test_setting = True
        return test_setting

    @staticmethod
    def is_loihi2_available() -> bool:
        """Checks if Loihi 2 is available and can be accessed."""

        is_loihi2 = False
        is_slurm = False
        is_nxcore = False

        # Check if SLURM is running
        if os.getenv("SLURM") == "1":
            is_slurm = True

            # Check if Loihi2 is available
            sinfo = subprocess.run("sinfo",  # nosec # noqa: S603, S607
                                   stdout=subprocess.PIPE).stdout.decode(
                'utf-8')
            for line in sinfo.split("\n"):
                if line.startswith(("oheogulch", "kp")):
                    is_loihi2 = True

        # Check if NxSDK is working
        if importlib.util.find_spec("nxcore") is not None:
            is_nxcore = True

        if is_loihi2 and is_slurm and is_nxcore:
            return True
        else:
            return False
