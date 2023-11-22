import os
import logging
import wandb
import sys
import submitit
import torch
from torch import distributed as dist
from dataclasses import dataclass, field, is_dataclass
import pathlib
from submitit import SlurmExecutor
from simple_parsing import ArgumentParser, subgroups, Serializable
import typing as tp


# dataclass built in asdict doesn't work with submitit's serialization
# as a hack, we use this function instead
def asdict(dataclass):
    out = {}
    for k, v in dataclass.__dict__.items(): 
        if is_dataclass(v): 
            out[k] = asdict(v)
        else:
            out[k] = v
    return out


def slurm_checkpoint_dir():
    """
    Returns the path to the slurm checkpoint directory if running on a slurm cluster,
    otherwise returns None. (This function is designed to work on the vector cluster)
    """
    import os

    if "SLURM_JOB_ID" not in os.environ:
        return None
    return os.path.join("/checkpoint", os.environ["USER"], os.environ["SLURM_JOB_ID"])


def generate_experiment_name():
    """
    Generates a fun experiment name.
    """
    from coolname import generate_slug
    from datetime import datetime

    return f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}-{generate_slug(2)}'


def generate_exp_dir(name=None, project=None, group=None):
    """
    Generates a directory name for the experiment.
    """
    exp_name = name or generate_experiment_name()
    exp_dir = "logs"
    if project is not None:
        exp_dir = os.path.join(exp_dir, project)
    if group is not None:
        exp_dir = os.path.join(exp_dir, group)
    exp_dir = os.path.join(exp_dir, exp_name)
    return exp_dir


# TODO: move this to the BasicDDPExperiment class
def basic_ddp_experiment_setup(
    exp_dir, group=None, config_dict=None, wandb_project=None, resume=True, debug=False
):
    dist_env = submitit.helpers.TorchDistributedEnvironment().export()
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # setup logging for each process
    file_handler = logging.FileHandler(
        os.path.join(exp_dir, f"out_rank{dist_env.rank}.log")
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO, handlers=[file_handler, stdout_handler], force=True
    )

    # also log tracebacks with excepthook
    def excepthook(type, value, tb):
        logging.error("Uncaught exception: {0}".format(str(value)))
        import traceback

        traceback.print_tb(tb, file=open(os.path.join(exp_dir, "out.log"), "a"))
        logging.error(f"Exception type: {type}")
        sys.__excepthook__(type, value, tb)

    sys.excepthook = excepthook

    logging.info(f"master: {dist_env.master_addr}:{dist_env.master_port}")
    logging.info(f"rank: {dist_env.rank}")
    logging.info(f"world size: {dist_env.world_size}")
    logging.info(f"local rank: {dist_env.local_rank}")
    logging.info(f"local world size: {dist_env.local_world_size}")
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logging.info(f"{torch.cuda.is_available()=}")
    logging.info(f"{torch.cuda.device_count()=}")
    logging.info(f"{torch.cuda.current_device()=}")
    logging.info(f"{torch.cuda.get_device_name()=}")
    logging.info(f"{torch.cuda.get_device_capability()=}")
    logging.info("initializing process group")
    # Using the (default) env:// initialization method
    torch.distributed.init_process_group(backend="nccl")
    torch.distributed.barrier()
    logging.info("process group initialized")

    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    if dist_env.rank == 0:
        if not os.path.exists(os.path.join(exp_dir, "checkpoints")):
            if slurm_checkpoint_dir() is not None:
                # sym link slurm checkpoints dir to local checkpoints dir
                os.symlink(slurm_checkpoint_dir(), os.path.join(exp_dir, "checkpoints"))
            else:
                os.makedirs(os.path.join(exp_dir, "checkpoints"))

        import json

        if config_dict is not None:
            json.dump(
                config_dict, open(os.path.join(exp_dir, "config.json"), "w"), indent=4
            )

        if resume and "wandb_id" in os.listdir(exp_dir):
            wandb_id = open(os.path.join(exp_dir, "wandb_id")).read().strip()
            logging.info(f"Resuming wandb run {wandb_id}")
        else:
            wandb_id = wandb.util.generate_id()
            open(os.path.join(exp_dir, "wandb_id"), "w").write(wandb_id)

        wandb.init(
            project=wandb_project if not debug else "debug",
            group=group,
            config=config_dict,
            resume="allow",
            name=os.path.basename(exp_dir),
            id=wandb_id,
            dir=ckpt_dir,
        )
        logging.info(f"wandb run: {wandb.run.name}")
        logging.info(f"wandb dir: {wandb.run.dir}")
        logging.info(f"wandb id: {wandb.run.id}")
        logging.info(f"wandb url: {wandb.run.get_url()}")

    dist.barrier()
    return ckpt_dir, dist_env


@dataclass
class SubmititJobSubmissionConfig:
    """Configuration for running the job in a slurm cluster using submitit."""

    timeout_min: int = 60 * 2
    slurm_gres: str = "gpu:1"
    mem_gb: int = 16
    cpus_per_task: int = 16
    slurm_qos: tp.Literal["normal", "m2", "m3", "m4"] = "m2"


@dataclass
class LocalJobSubmissionConfig:
    """Configuration for running the job locally"""


@dataclass
class BasicExperimentConfig(Serializable):
    """
    Basic configuration for the experiment.
    """

    exp_dir: str = None
    name: str = None
    group: str = None
    project: str = None
    entity: str = None
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    cluster: SubmititJobSubmissionConfig | LocalJobSubmissionConfig = subgroups(
        {"slurm": SubmititJobSubmissionConfig, "local": LocalJobSubmissionConfig},
        default="local",
    )

    def __post_init__(self):
        if self.name is None:
            self.name = generate_experiment_name()
        if self.exp_dir is None:
            self.exp_dir = generate_exp_dir(self.name, self.project, self.group)


class BasicExperiment:
    """
    Base class for an experiment. Handles boilerplate setup such
    as logging, experiment and checkpoint directory setup, and
    enables automatic argument parsing and submission to a cluster.

    Example usage:
    ```
    @dataclass
    class MyExperimentConfig(BasicExperimentConfig):
        # add your config options here
        pass

    class MyExperiment(BasicExperiment):
        config_class = MyExperimentConfig
        config: MyExperimentConfig

        def setup(self):
            super().setup()
            # add your setup code here

        def __call__(self):
            # implement your experiment here
            pass

    if __name__ == '__main__':
        MyExperiment.submit() # handles argument parsing and submission to cluster
    """

    config_class = BasicExperimentConfig
    config: BasicExperimentConfig

    def __init__(self, config: config_class):
        self.config = config

    def __call__(self):
        """
        Runs the experiment.
        """
        raise NotImplementedError

    def setup(self):
        os.makedirs(self.config.exp_dir, exist_ok=True)
        if not os.path.exists(os.path.join(self.config.exp_dir, "config.yaml")):
            with open(os.path.join(self.config.exp_dir, "config.yaml"), "w") as f:
                import yaml
                yaml.dump(asdict(self.config), f)
        
        if not os.path.exists(os.path.join(self.config.exp_dir, "checkpoints")):
            if slurm_checkpoint_dir() is not None:
                # sym link slurm checkpoints dir to local checkpoints dir
                os.symlink(
                    slurm_checkpoint_dir(),
                    os.path.join(self.config.exp_dir, "checkpoints"),
                )
            else:
                os.makedirs(os.path.join(self.config.exp_dir, "checkpoints"))
        ckpt_dir = os.path.join(self.config.exp_dir, "checkpoints")

        stdout_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(os.path.join(self.config.exp_dir, "out.log"))
        logging.basicConfig(
            level=logging.INFO, handlers=[stdout_handler, file_handler], force=True
        )

        # also log tracebacks with excepthook
        def excepthook(type, value, tb):
            logging.error("Uncaught exception: {0}".format(str(value)))
            import traceback

            traceback.print_tb(
                tb, file=open(os.path.join(self.config.exp_dir, "out.log"), "a")
            )
            logging.error(f"Exception type: {type}")
            sys.__excepthook__(type, value, tb)

        sys.excepthook = excepthook

        if self.config.resume and "wandb_id" in os.listdir(self.config.exp_dir):
            wandb_id = (
                open(os.path.join(self.config.exp_dir, "wandb_id")).read().strip()
            )
            logging.info(f"Resuming wandb run {wandb_id}")
        else:
            wandb_id = wandb.util.generate_id()
            open(os.path.join(self.config.exp_dir, "wandb_id"), "w").write(wandb_id)

        if not self.config.use_wandb:
            os.environ["WANDB_MODE"] = "disabled"

        wandb.init(

            project=self.config.project
            if not self.config.debug
            else f"{self.config.project}-debug",
            group=self.config.group,
            config=asdict(self.config),
            resume="allow",
            name=os.path.basename(self.config.exp_dir),
            id=wandb_id,
            dir=ckpt_dir,
        )
        self.ckpt_dir = ckpt_dir

    def checkpoint(self):
        """
        Handles checkpointing the experiment when running on a cluster
        using submitit. This method is called by submitit when the job is preempted or times out.
        """
        logging.info(f"Handling Preemption or timeout!")
        from submitit.helpers import DelayedSubmission

        new_job = self.__class__(self.config)
        logging.info(f"Resubmitting myself.")
        return DelayedSubmission(new_job)

    @classmethod
    def submit(cls):
        from simple_parsing import ArgumentParser

        parser = ArgumentParser()
        parser.add_arguments(cls.config_class, dest="config")
        parser.add_argument(
            "--config_path", type=str, default=None, help="Path to a config file."
        )
        args = parser.parse_args()
        if args.config_path is not None:
            cfg = cls.config_class.load_yaml(args.config_path)
        else:
            cfg = args.config

        job = cls(cfg)
        if isinstance(cfg.cluster, LocalJobSubmissionConfig):
            job()

        elif isinstance(cfg.cluster, SubmititJobSubmissionConfig):
            os.makedirs(cfg.exp_dir, exist_ok=True)
            executor = submitit.AutoExecutor(
                folder=os.path.join(cfg.exp_dir, "submitit_logs"),
                slurm_max_num_timeout=10,
            )

            executor.update_parameters(**asdict(cfg.cluster))
            job = executor.submit(job)
            print(f"Submitted job: {job.job_id}")
            print(f"Outputs: {job.paths.stdout}")
            print(f"Errors: {job.paths.stderr}")


# TODO: finish and test this
class BasicDDPExperiment(BasicExperiment):
    """"""

    def setup(self):
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()
        self.dist_env = dist_env
        exp_dir = self.config.exp_dir

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # setup logging for each process
        file_handler = logging.FileHandler(
            os.path.join(exp_dir, f"out_rank{dist_env.rank}.log")
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        logging.basicConfig(
            level=logging.INFO, handlers=[file_handler, stdout_handler], force=True
        )

        # also log tracebacks with excepthook
        def excepthook(type, value, tb):
            logging.error("Uncaught exception: {0}".format(str(value)))
            import traceback

            traceback.print_tb(tb, file=open(os.path.join(exp_dir, "out.log"), "a"))
            logging.error(f"Exception type: {type}")
            sys.__excepthook__(type, value, tb)

        sys.excepthook = excepthook

        logging.info(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        logging.info(f"rank: {dist_env.rank}")
        logging.info(f"world size: {dist_env.world_size}")
        logging.info(f"local rank: {dist_env.local_rank}")
        logging.info(f"local world size: {dist_env.local_world_size}")
        logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        logging.info(f"{torch.cuda.is_available()=}")
        logging.info(f"{torch.cuda.device_count()=}")
        logging.info(f"{torch.cuda.current_device()=}")
        logging.info(f"{torch.cuda.get_device_name()=}")
        logging.info(f"{torch.cuda.get_device_capability()=}")
        logging.info("initializing process group")
        # Using the (default) env:// initialization method
        torch.distributed.init_process_group(backend="nccl")
        torch.distributed.barrier()
        logging.info("process group initialized")

        ckpt_dir = os.path.join(exp_dir, "checkpoints")

        if dist_env.rank == 0:
            if not os.path.exists(os.path.join(exp_dir, "checkpoints")):
                if slurm_checkpoint_dir() is not None:
                    # sym link slurm checkpoints dir to local checkpoints dir
                    os.symlink(
                        slurm_checkpoint_dir(), os.path.join(exp_dir, "checkpoints")
                    )
                else:
                    os.makedirs(os.path.join(exp_dir, "checkpoints"))

            with open(os.path.join(self.config.exp_dir, "config.yaml"), "w") as f:
                self.config.dump_yaml(f)

        if dist_env.rank != 0 or not self.config.use_wandb:
            os.environ["WANDB_MODE"] = "disabled"

        if self.config.resume and "wandb_id" in os.listdir(exp_dir):
            wandb_id = open(os.path.join(exp_dir, "wandb_id")).read().strip()
            logging.info(f"Resuming wandb run {wandb_id}")
        else:
            wandb_id = wandb.util.generate_id()
            open(os.path.join(exp_dir, "wandb_id"), "w").write(wandb_id)

        wandb.init(
            project=self.config.project if not self.config.debug else "debug",
            group=self.config.group,
            config=asdict(self.config),
            resume="allow",
            name=os.path.basename(exp_dir),
            id=wandb_id,
            dir=ckpt_dir,
        )
        logging.info(f"wandb run: {wandb.run.name}")
        logging.info(f"wandb dir: {wandb.run.dir}")
        logging.info(f"wandb id: {wandb.run.id}")
        logging.info(f"wandb url: {wandb.run.get_url()}")

        dist.barrier()
