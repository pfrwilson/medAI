import os


def scratch_dir():
    return '/scratch/ssd004/scratch/pwilson/'


def slurm_checkpoint_dir():
    return os.path.join('/checkpoint', os.environ['USER'], os.environ['SLURM_JOB_ID'])
