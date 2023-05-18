export slurmscript=$OAK/projects/astrock/2021_common/scripts/slurm
alias sourcenv='[ -f ./environment.sh ] && { source ./environment.sh; echo "Starting project $PROJECT_NAME"; } || { echo "Setup your environment before running."; }'
export flag1c='--mem 50G -p normal,menon -c 1'
export flag8c='--mem 50G -p normal,menon -c 8'
export flag1g='--mem 50G -p menon,normal,gpu -c 8 -G 1'
export flag2g='--mem 50G -p menon,normal,gpu -c 8 -G 2 -N 1'
export flag4g='--mem 50G -p menon,normal,gpu -c 8 -G 4 -N 1'
export flagtime='10:00:00'

alias srm='python $slurmscript/rmsubmit.py --time=$flagtime $flag1c'
alias smv='python $slurmscript/mvsubmit.py --time=$flagtime $flag1c'

alias debug1c='[ -f ./environment.sh ] && { sourcenv; srun --time=$flagtime $flag1c --pty bash; } || { echo "Setup your environment before running."; }'
alias debug8c='[ -f ./environment.sh ] && { sourcenv; srun --time=$flagtime $flag8c --pty bash; } || { echo "Setup your environment before running."; }'
alias debug1g='[ -f ./environment.sh ] && { sourcenv; srun --time=$flagtime $flag1g --pty bash; } || { echo "Setup your environment before running."; }'
alias debug2g='[ -f ./environment.sh ] && { sourcenv; srun --time=$flagtime $flag2g --gpu_cmode=shared --pty bash; } || { echo "Setup your environment before running."; }'
alias debug4g='[ -f ./environment.sh ] && { sourcenv; srun --time=$flagtime $flag4g --gpu_cmode=shared --pty bash; } || { echo "Setup your environment before running."; }'

alias submit1c='python $slurmscript/pysubmit.py --time=$flagtime $flag1c'
alias submit8c='python $slurmscript/pysubmit.py --time=$flagtime $flag8c'
alias submit1g='python $slurmscript/pysubmit.py --time=$flagtime $flag1g'
alias submit4g='python $slurmscript/pysubmit.py --time=$flagtime $flag4g --Gshared'

alias submit4g='python $slurmscript/pysubmit.py $flag4g'