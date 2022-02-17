# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import os


class Utils():

    @staticmethod
    def get_env_test_setting(env: str):
        env_test_setting = os.environ.get(env)
        test_setting = False
        if env_test_setting == "1":
            test_setting = True
        return test_setting
