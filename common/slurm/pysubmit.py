#!/usr/bin/python

import argparse, os
import subprocess

def main(args):
    job = os.path.splitext(args.s[0])[0].replace('/', '-')
    tmp_dir = '$TMP_PATH/log'
    if args.pepochs is None and args.psteps is None:
        out = '{}/{}-%j.out'.format(tmp_dir, job)
        array_shflag = ''
        array_pyflag = ''
    elif not args.pepochs is None and not args.psteps is None:
        raise ValueError("Can't run both in parallel for epochs (i.e. --pepochs) and for steps (i.e. --psteps)")
    else:
        if not args.pepochs is None:
            array_pyflag = ' --epochs'
            l = args.pepochs
        else:
            array_pyflag = ' --steps'
            l = args.psteps
        print(l)
        shift = min(l)
        shift = 0 if shift > 0 else shift
        print(type(shift), shift)
        out = '{}/{}-%A-%a{}.out'.format(tmp_dir, job, shift) if shift < 0 else '{}/{}-%A-%a.out'.format(tmp_dir, job)
        array_shflag = ' --array={}'.format(','.join(['{:d}'.format(x-shift) for x in l]))
        array_pyflag = '{} \$((\${{SLURM_ARRAY_TASK_ID}}{}))'.format(array_pyflag, ' - {:d}'.format(-shift) if shift < 0 else '')
    gflags = ' -G {:d}{}'.format(args.G, ' --gpu_cmode=shared' if args.Gshared else '') if args.G>0 else ''
    flags = '-J {}{} -o {} -N {} -c {:d}{} -t {} --mem={} -p {} {}'.format(job, array_shflag, out, args.N, args.c, gflags, args.t, args.m, ','.join(args.p), args.d)
    cmd = '[ -f ./environment.sh ] && {{ source ./environment.sh; echo "Starting project $PROJECT_NAME"; sbatch {} --wrap="srun python {}{}"; }} || {{ echo "Setup your environment before running."; }}'.format(flags, ' '.join(args.s), array_pyflag)
    os.system(cmd)

class ValidatePartition(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        valid_p = ['menon','owners','normal','dev','long','gpu','bigmem']
        for p in values:
            if p not in valid_p:
                raise ValueError('invalid partition {}\nValid partition are: {}'.format(p, ','.join(valid_p)))
        setattr(args, self.dest, values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'pysubmit submits python scripts to the SLURM queue on the Sherlock Cluster', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('s', metavar = 'S', type = lambda x: str(x).split(" "), help = 'python script (arguments between "", i.e. "SCRIPT ARGS")')
    parser.add_argument('-p', metavar = 'P', type = lambda x: str(x).split(","), default = ['menon'], action = ValidatePartition,  help = 'sets the slurm partition, default=menon, options=menon,owners,normal,dev,long,gpu,bigmem')
    parser.add_argument('-t', '--time', dest = 't', metavar = 'T', type = str, nargs = "?", default = '02:00:00', help = 'sets the job time alloted, default=02:00:00')
    parser.add_argument('-m', '--mem', dest = 'm', metavar = 'M', type = str, nargs = "?", default = '32G', help = 'sets the memory for the job, default=32G')
    parser.add_argument('-N', metavar = 'N', type = int, nargs = "?", default = 1, help = 'sets the number of nodes for the job, default=1')
    parser.add_argument('-c', metavar = 'C', type = int, nargs = "?", default = 8, help = 'sets the cpus for the job, default=8')
    parser.add_argument('-G', metavar = 'G', type = int, nargs = "?", default = 0, help = 'sets the gpus for the job, default=0')
    parser.add_argument('--pepochs', metavar = 'E', type = int, nargs = "+", required = False, default = None, help = 'epochs to run in parallel')
    parser.add_argument('--psteps', metavar = 'S', type = int, nargs = "+", required = False, default = None, help = 'steps to run in parallel')
    parser.add_argument('--Gshared', action = 'store_true', help = 'sets the gpu cmode to shared')
    parser.add_argument('-d', metavar = 'D', type = lambda x: '' if x == '' else '--dependency='+str(x), nargs = "?", default = '', help = '''sets the dependency, default=none,
after:jobid[:jobid...]  job can begin after the specified jobs have started,
afterany:jobid[:jobid...] job can begin after the specified jobs have terminated,
afternotok:jobid[:jobid...] job can begin after the specified jobs have failed,
afterok:jobid[:jobid...]  job can begin after the specified jobs have run to completion,
singleton jobs can begin execution after previously launched jobs with the same name and user have ended.''')

	
    args = parser.parse_args()
    main(args)
