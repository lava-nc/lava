import os
import platform
import multiprocessing
import numpy
import subprocess  # nosec
import sys


def build_msg_lib() -> bool:
    pure_py_env = os.getenv("LAVA_PURE_PYTHON", 0)
    system_name = platform.system().lower()
    if system_name != "linux":
        return False
    return int(pure_py_env) == 0


class CMake:
    def __init__(self, sourcedir, targetdir):
        self.sourcedir = os.path.abspath(sourcedir)
        self.targetdir = os.path.abspath(targetdir)
        self.env = os.environ.copy()
        self.from_poetry = self._check_poetry()
        self.from_cd_action = self._check_cd_action()
        self.cmake_command = ["poetry", "run", "cmake"] \
            if self.from_cd_action and self.from_poetry else ["cmake"]
        self.cmake_args = []
        self.build_args = []

    def _check_poetry(self):
        exec_code = self.env.get('_', "").rsplit('/')[-1]
        if exec_code == 'poetry':
            return True
        return False

    def _check_cd_action(self):
        event_name = self.env.get('GITHUB_EVENT_NAME', '')
        return event_name == 'workflow_dispatch'

    def _set_cmake_path(self):
        self.temp_path = os.path.join(os.path.abspath(""), "build")
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

    def _set_cmake_args(self):
        debug = int(os.environ.get("DEBUG", 0))
        cfg = "Debug" if debug else "Release"
        if self.from_cd_action and self.from_poetry:
            python_env = subprocess.check_output(["poetry", "env", "info", "-p"]) \
                             .decode().strip() + "/bin/python3"  # nosec # noqa
            numpy_include_dir = subprocess.check_output(["poetry", "run",  # nosec # noqa
                "python3", "-c", "import numpy; print(numpy.get_include())"]).decode().strip()  # nosec # noqa
        else:
            python_env = sys.executable
            numpy_include_dir = numpy.get_include()
        self.cmake_args += [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={self.targetdir}",
            f"-DPYTHON_EXECUTABLE={python_env}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        if "CMAKE_ARGS" in os.environ:
            self.cmake_args += [item for item in
                                os.environ["CMAKE_ARGS"].split(" ") if item]
        # Set numpy include header to cpplib
        self.cmake_args += [
            f"-DNUMPY_INCLUDE_DIRS={numpy_include_dir}"]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            self.parallel = multiprocessing.cpu_count()
            self.build_args += [f"-j{self.parallel}"]

    def run(self):
        self._set_cmake_path()
        self._set_cmake_args()
        subprocess.check_call([*self.cmake_command, self.sourcedir] + self.cmake_args, cwd=self.temp_path, env=self.env)  # nosec # noqa
        subprocess.check_call([*self.cmake_command, "--build", "."] + self.build_args, cwd=self.temp_path, env=self.env)  # nosec # noqa


if __name__ == '__main__':
    base_runtime_path = "src/lava/magma/runtime/"
    sourcedir = f"{base_runtime_path}_c_message_infrastructure"
    targetdir = f"{base_runtime_path}message_infrastructure"
    if build_msg_lib():
        cmake = CMake(sourcedir, targetdir)
        cmake.run()
