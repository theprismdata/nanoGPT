# Configuration for tax law instruction tuning on nanoGPT
# This config is optimized for Korean tax law instruction-following tasks

out_dir = 'out-tax-law'
eval_interval = 200  # evaluate frequently for instruction data
eval_iters = 50
log_interval = 10

# Save checkpoints when validation improves
always_save_checkpoint = False

# MLflow logging
mlflow_log = True
mlflow_experiment_name = 'tax-law-instruction'
mlflow_run_name = 'nanogpt-tax-law'

# Dataset configuration
dataset = 'tax-law-instruction'
data_dir = 'training-data/tax-law-instruction'  # Use prepared tax law data
gradient_accumulation_steps = 1  # reduce memory usage
batch_size = 4  # smaller batch size for Mac M4
block_size = 512  # shorter context to save memory

# Model architecture - smaller size for Mac M4 memory constraints
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1  # some regularization for instruction tuning

# Learning rate and training schedule
learning_rate = 2e-4  # lower learning rate for instruction tuning
max_iters = 2000  # fewer iterations for testing
lr_decay_iters = 2000
min_lr = 2e-5  # minimum learning rate
beta2 = 0.95

# Warmup
warmup_iters = 100  # shorter warmup for testing

# For Mac M4 optimization
device = 'mps'
compile = False

# Initialize from scratch or from a foundation model
init_from = 'scratch'  # Change to 'resume' or model path if you have a pretrained model

# Additional settings for instruction tuning
weight_decay = 1e-2
beta1 = 0.9
grad_clip = 1.0
decay_lr = True
