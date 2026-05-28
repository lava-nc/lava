"""Unit Tests for RunConfigs.

"""

import unittest
import logging

from lava.magma.core.run_configs import (
    Loihi2SimCfg, Loihi2HwCfg, GeneralRunCfg
)
from lava.magma.core.resources import CPU, Loihi2System, LMT

class TestGeneralRunConfig(unittest.TestCase):
    """Unit tests for the GeneralRunConfig class."""
    def test_new(self):
        """Test that the `__new__` function returns the correct `RunConfig`."""

        cpu_run_cfg = GeneralRunCfg(hardware=CPU)
        self.assertIsInstance(cpu_run_cfg, Loihi2SimCfg)

        loihi2_run_cfg = GeneralRunCfg(hardware=Loihi2System)
        self.assertIsInstance(loihi2_run_cfg, Loihi2HwCfg)

        with self.assertRaises(KeyError):
            GeneralRunCfg(hardware=LMT)

    def test_cpu_config(self):
        """Test instantiation of CPU run configs."""
        hardware = CPU
        run_cfg = GeneralRunCfg(hardware=hardware)

        self.assertEqual(run_cfg.log.level, logging.WARNING)
        self.assertEqual(run_cfg.custom_sync_domains, [])
        self.assertEqual(run_cfg.select_tag, None)
        self.assertEqual(run_cfg.select_sub_proc_model, False)
        self.assertEqual(run_cfg.exception_proc_model_map, {})

        # test with kwargs
        run_cfg = GeneralRunCfg(
            hardware=hardware,
            select_tag='fixed_pt',
            select_sub_proc_model=True,
            loglevel=logging.INFO
        )
        self.assertEqual(run_cfg.log.level, logging.INFO)
        self.assertEqual(run_cfg.custom_sync_domains, [])
        self.assertEqual(run_cfg.select_tag, 'fixed_pt')
        self.assertEqual(run_cfg.select_sub_proc_model, True)
        self.assertEqual(run_cfg.exception_proc_model_map, {})

    def test_loihi2_config(self):
        """Test instantiation of Loihi2 run configs."""
        hardware = Loihi2System
        run_cfg = GeneralRunCfg(hardware=hardware)

        self.assertEqual(run_cfg.log.level, logging.WARNING)
        self.assertEqual(run_cfg.custom_sync_domains, [])
        self.assertEqual(run_cfg.select_tag, None)
        self.assertEqual(run_cfg.select_sub_proc_model, False)
        self.assertEqual(run_cfg.exception_proc_model_map, {})
        self.assertEqual(run_cfg.callback_fxs, [])
        self.assertEqual(run_cfg.embedded_allocation_order, 1)

        # test with kwargs
        run_cfg = GeneralRunCfg(
            hardware=hardware,
            select_sub_proc_model=True,
            loglevel=logging.INFO
        )
        self.assertEqual(run_cfg.log.level, logging.INFO)
        self.assertEqual(run_cfg.custom_sync_domains, [])
        self.assertEqual(run_cfg.select_tag, None)
        self.assertEqual(run_cfg.select_sub_proc_model, True)
        self.assertEqual(run_cfg.exception_proc_model_map, {})
        self.assertEqual(run_cfg.callback_fxs, [])
        self.assertEqual(run_cfg.embedded_allocation_order, 1)
