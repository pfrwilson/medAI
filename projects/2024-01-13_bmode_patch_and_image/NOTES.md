___
# Sat, Jan 13 2024: 

Created this repository by porting over from a previous one. Currently attempting to reproduce the MEDSAM model finetune performance.
- needle region as valid loss region: 
    https://wandb.ai/pfrwilson/ultrasound_patch_ssl-debug/runs/qvws3ywk?workspace=user-pfrwilson
- prostate region as valid loss region
    https://wandb.ai/pfrwilson/ultrasound_patch_ssl-debug/runs/6k1ii04r?workspace=user-pfrwilson

Next step will be to implement the patch-based self-supervised learning and get a baseline performance 
level for this. We expect something in the range of mid-70s AUC. 

After this, we want to fuse the self-supervised feature extraction with a medsam-based feature extraction. Goal is to see some benefit here. 

***THOUGHTS*** - KFold cross-validation has been criticized by users and it might be good to stay with leave-one-center-out cross validation. For example, we could fix a center for all these baselines.  
___

# Tues Jan 16

So far we have not been able to reproduce good performance with VICReg on Bmode using the `train_ssl_patch_wise.py` script. However, we did observe the same good performance with medsam finetuning, sometimes achieving high-70's in terms of AUC. 
