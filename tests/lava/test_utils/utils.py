# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import os


class Utils():
    """Utility Class containing testing helper
    code that can be reused between tests
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
