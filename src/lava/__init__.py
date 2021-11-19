# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
try:
    __import__('pkg_resources').declare_namespace(__name__)
except:  # noqa E722
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
