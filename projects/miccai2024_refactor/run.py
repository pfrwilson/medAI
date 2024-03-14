from train_prostnfound import *
import submitit
import os

DEBUG=False

TEST_CENTER = "UVA"
VAL_SEED=0

data_cfg = BModeDataFactoryV1Config(
    test_center=TEST_CENTER,
    undersample_benign_ratio=6,
    remove_benign_cores_from_positive_patients=True,
    batch_size=4,
)

model_cfg = ProstNFoundConfig(
    prompts=[
        PromptOptions.age,
        PromptOptions.approx_psa_density,
        PromptOptions.psa,
        PromptOptions.sparse_cnn_patch_features_rf
    ],
    sam_backbone=BackboneOptions.medsam,
    sparse_cnn_backbone_path=f"/ssd005/projects/exactvu_pca/checkpoint_store/miccai2024_ProFound_PatchSSLWeights/{TEST_CENTER}_patch_ssl_rf{VAL_SEED}.pth", 
    pool_patch_features='max', 
    pos_embed_cnn_patch=False,    
)

wandb_config = WandbConfig(project='miccai2024_repro', name="debug" if DEBUG else None)

loss_config = CancerDetectionValidRegionLossConfig(loss_pos_weight=2)

optim_config = OptimizerConfig(
    main_lr=1e-5, 
    encoder_lr=1e-5, 
    cnn_lr=1e-6, 
    cnn_frozen_epochs=20, 
    cnn_warmup_epochs=3,    
)

MISSING="???"

args = Args(
    wandb=wandb_config,
    data=data_cfg,
    model=model_cfg,
    optimizer=optim_config,
    losses=[loss_config],
    loss_weights=[1.0],
    epochs=35, 
    cutoff_epoch=None, 
    test_every_epoch=True, 
    accumulate_grad_steps=2, 
    run_test=True, 
    use_amp=True, 
    device='cuda' if not DEBUG else 'cpu', 
    checkpoint_dir=MISSING, # set at runtime
    exp_dir=MISSING, # set at runtime
    seed=42
)


class Main:
    def __init__(self, args: Args):
        self.args = args

    def __call__(self):
        SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
        os.environ["TQDM_MININTERVAL"] = "30"
        os.environ["WANDB_RUN_ID"] = f"{SLURM_JOB_ID}"
        os.environ["WANDB_RESUME"] = "allow"
        CKPT_DIR = f'/checkpoint/{os.environ["USER"]}/{SLURM_JOB_ID}'
        
        if self.args.checkpoint_dir == MISSING: 
            self.args.checkpoint_dir = CKPT_DIR
        if self.args.exp_dir == MISSING: 
            self.args.exp_dir = CKPT_DIR

        experiment = Experiment(args)
        experiment.run()

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(Main(self.args))


if not DEBUG: 

    executor = submitit.AutoExecutor(folder="logs", slurm_max_num_timeout=10)
    if PromptOptions.sparse_cnn_patch_features_rf in args.model.prompts: 
        mem="64G"
    else: 
        mem="32G"
    executor.update_parameters(
        slurm_mem=mem,
        slurm_gres='gpu:a40:1', 
        slurm_time = "8:00:00", 
        cpus_per_task=16,
        slurm_qos='m2', 
        stderr_to_stdout=True,
    )

    job = executor.submit(Main(args))
    print(f"Submitted job {job.job_id}")
    print(f"Logs at {job.paths.stdout}")

else: 
    args.data.batch_size = 1
    Experiment(args).run()