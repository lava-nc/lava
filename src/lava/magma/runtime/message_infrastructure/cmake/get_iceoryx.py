import sys
import os
import subprocess
import multiprocessing

package = "iceoryx"
url = "https://github.com/eclipse-iceoryx/iceoryx.git"
tag = "v2.0.2"


def main():
    if len(sys.argv) < 3:
        return

    dst_path = sys.argv[1]
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    clone_path = sys.argv[2]
    clone_path = os.path.join(clone_path, package)
    subprocess.call(["git", "clone", "--depth", "1", "--branch", tag, url, clone_path])  # nosec # noqa
    # build iceoryx
    subprocess.call(["cmake", "-Bbuild", "-Hiceoryx_meta", f"-DCMAKE_INSTALL_PREFIX={os.path.abspath(dst_path)}"],  # nosec # noqa
                    cwd=os.path.abspath(clone_path))
    # install iceoryx
    subprocess.call(["cmake", "--build", "build", "--target", "install", f"-j{multiprocessing.cpu_count()}"],  # nosec # noqa
                    cwd=os.path.abspath(clone_path))


if __name__ == "__main__":
    main()
