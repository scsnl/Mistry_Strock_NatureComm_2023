# ------------------
# PROJECT PATHS
# ------------------
export PROJECT_PATH=.
export FIG_PATH=${PROJECT_PATH}/figures
export DATA_PATH=${PROJECT_PATH}/data

# ------------------
# PYTHON SETUP
# ------------------
ml system libnvidia-container
source $GROUP_HOME/python_envs/pyenv/activate.sh
export NN_COMMON=$PROJECT_PATH/common
export PYTHONPATH=$NN_COMMON:$PYTHONPATH
