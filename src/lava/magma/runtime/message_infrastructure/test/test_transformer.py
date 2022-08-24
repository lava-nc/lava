# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from MessageInfrastructurePywrapper import (VirtualPortTransformer, 
                                            IdentityTransformer)

class TestIdentityTransformer(unittest.TestCase):
    def test_init(self) -> None:
        """Tests the initialization of an IdentityTransformer."""
        it = IdentityTransformer()
        self.assertIsInstance(it, IdentityTransformer)

    def test_transform(self) -> None:
        """Tests whether the transformation is the identity transformation."""
        it = IdentityTransformer()
        data = np.array(5)
        self.assertEqual(it.transform(data), data)
