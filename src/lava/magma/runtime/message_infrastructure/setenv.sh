SCRIPTPATH=$(cd `dirname -- $0` && pwd)
export PYTHONPATH="${SCRIPTPATH}/build:$PYTHONPATH"
export LD_LIBRARY_PATH="${SCRIPTPATH}/build:$LD_LIBRARY_PATH"
