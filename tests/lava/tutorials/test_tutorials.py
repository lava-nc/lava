# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""Test tutorials"""

import glob
import os
import platform
import subprocess  # noqa: S404
import tempfile
import unittest
from test import support

import nbformat

import lava
import tutorials


class TestTutorials(unittest.TestCase):
    """Export notebook, execute to check for errors"""

    system_name = platform.system().lower()

    def _execute_notebook(self, base_dir, path):
        """Execute a notebook via nbconvert and collect output.

        Parameters
        ----------
        base_dir : str
            notebook search directory
        path : str
            path to notebook

        Returns
        -------
        Tuple
            (parsed nb object, execution errors)
        """

        cwd = os.getcwd()
        dir_name, notebook = os.path.split(path)
        try:
            os.chdir(base_dir + "/" + dir_name)

            env = os.environ.copy()
            module_path = [lava.__path__.__dict__["_path"][0]]
            # Path: module path + parent dir of module + existing PYTHONPATH
            module_path.extend(
                [os.path.dirname(module_path[0]), env.get("PYTHONPATH", "")])
            env["PYTHONPATH"] = ":".join(module_path)

            with tempfile.NamedTemporaryFile(mode="w+t", suffix=".ipynb") \
                    as fout:
                args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--ExecutePreprocessor.timeout=-1",
                        "--output", fout.name, notebook]
                subprocess.check_call(args, env=env)  # noqa: S603

                fout.seek(0)
                nb = nbformat.read(fout, nbformat.current_nbformat)

            errors = []
            for cell in nb.cells:
                if 'outputs' in cell:
                    for output in cell['outputs']:
                        if output.output_type == 'error':
                            errors.append(output)

        except Exception as e:
            nb = None
            errors = str(e)
        finally:
            os.chdir(cwd)

        return nb, errors

    def _run_notebook(self, notebook, e2e_tutorial=False):
        """Run a specific notebook

        Parameters
        ----------
        notebook : str
            name of notebook to run
        e2e_tutorial : bool, optional
            end to end tutorial, by default False
        """
        cwd = os.getcwd()
        tutorials_directory = ""

        if not e2e_tutorial:
            tutorials_directory = \
                os.path.realpath(tutorials.__path__.__dict__["_path"][0]
                                 + "/in_depth")
        else:
            tutorials_directory = \
                os.path.realpath(tutorials.__path__.__dict__["_path"][0]
                                 + "/end_to_end")

        os.chdir(tutorials_directory)

        errors_record = {}

        try:
            glob_pattern = "**/{}".format(notebook)
            discovered_notebooks = sorted(
                glob.glob(glob_pattern, recursive=True))

            self.assertTrue(len(discovered_notebooks) != 0,
                            "Notebook not found. Input to function {}"
                            .format(notebook))

            for notebook_name in discovered_notebooks:
                nb, errors = self._execute_notebook(
                    str(tutorials_directory),
                    notebook_name
                )
                errors_joined = "\n".join(errors) if isinstance(
                    errors, list) else errors
                if errors:
                    errors_record[notebook_name] = (errors_joined, nb)

            self.assertFalse(errors_record,
                             "Failed to execute Jupyter Notebooks \
                                 with errors: \n {}".format(errors_record))
        finally:
            os.chdir(cwd)

    @unittest.skipIf(system_name != "linux", "Tests work on linux")
    def test_end_to_end_01_mnist(self):
        """Test tutorial end to end 01 mnist."""
        self._run_notebook(
            "tutorial01_mnist_digit_classification.ipynb",
            e2e_tutorial=True
        )

    @unittest.skip("Tutorial is text only and does not contain code")
    def test_in_depth_01_install_lava(self):
        """Test tutorial in depth install lava."""
        self._run_notebook(
            "tutorial01_installing_lava.ipynb"
        )

    @unittest.skipIf(system_name != "linux", "Tests work on linux")
    def test_in_depth_02_processes(self):
        """Test tutorial in depth processes"""
        self._run_notebook(
            "tutorial02_processes.ipynb"
        )

    @unittest.skipIf(system_name != "linux", "Tests work on linux")
    def test_in_depth_03_process_models(self):
        """Test tutorial in depth process models."""
        self._run_notebook(
            "tutorial03_process_models.ipynb"
        )

    @unittest.skipIf(system_name != "linux", "Tests work on linux")
    def test_in_depth_04__execution(self):
        """Test tutorial in depth execution."""
        self._run_notebook(
            "tutorial04_execution.ipynb"
        )

    @unittest.skipIf(system_name != "linux", "Tests work on linux")
    def test_in_depth_05__connect_processes(self):
        """Test tutorial in depth connect processes."""
        self._run_notebook(
            "tutorial05_connect_processes.ipynb"
        )

    @unittest.skip("Skip until \
        https://github.com/lava-nc/lava/issues/242 is fixed")
    def test_in_depth_06_hierarchical_processes(self):
        """Test tutorial in depth hierarchical processes."""
        self._run_notebook(
            "tutorial06_hierarchical_processes.ipynb"
        )

    @unittest.skipIf(system_name != "linux", "Tests work on linux")
    def test_in_depth_07_remote_memory_access(self):
        """Test tutorial in depth remote memory access."""
        self._run_notebook(
            "tutorial07_remote_memory_access.ipynb"
        )


if __name__ == '__main__':
    support.run_unittest(TestTutorials)
