import subprocess  # nosec

subprocess.check_call(["poetry", "run", "python3", "setup.py", "build"])