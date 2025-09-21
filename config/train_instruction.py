# Configuration for instruction tuning on nanoGPT
# This config is optimized for instruction-following tasks

out_dir = 'out-instruction'
eval_interval = 500  # evaluate less frequently for instruction data
eval_iters = 100
log_interval = 10

# Save checkpoints when validation improves
always_save_checkpoint = False

# MLflow logging
mlflow_log = True
mlflow_experiment_name = 'instruction-tuning'
mlflow_run_name = 'nanogpt-instruct'

# Dataset configuration
dataset = 'instruction'
data_dir = 'training-data/instruction'  # Use training-data folder
gradient_accumulation_steps = 4  # simulate larger batch size
batch_size = 16  # smaller batch size for instruction data
block_size = 512  # longer context for instruction-response pairs

# Model architecture - slightly larger for better instruction following
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1  # some regularization for instruction tuning

# Learning rate and training schedule
learning_rate = 3e-4  # lower learning rate for instruction tuning
max_iters = 10000  # more iterations for instruction learning
lr_decay_iters = 10000
min_lr = 3e-5  # minimum learning rate
beta2 = 0.95

# Warmup
warmup_iters = 500  # longer warmup for instruction data

# For Mac M4 optimization
# device = 'mps'
# compile = False
