# Foundation Model Training Configuration - SMALL
# Estimated parameters: 124M
# Training from scratch without any pretrained weights

# Output directory
out_dir = 'out-foundation-small'

# Evaluation settings
eval_interval = 2000
eval_iters = 200
log_interval = 100

# Checkpoint settings
always_save_checkpoint = True

# Logging (optional)
wandb_log = False  # Set to True if you want to track experiments
wandb_project = 'foundation-model'
wandb_run_name = 'foundation-small-v1'

# Data configuration
dataset = 'foundation'  # Prepare your foundation dataset
gradient_accumulation_steps = 8  # Simulate larger batch sizes
batch_size = 12
block_size = 1024

# Model architecture - SMALL Foundation Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # No dropout for pretraining
bias = False   # Remove bias for efficiency

# Optimizer settings (based on GPT-3 paper)
learning_rate = 0.0006
max_iters = 200000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 200000
min_lr = 5.9999999999999995e-05

# System settings (Mac M4 optimized)
device = 'mps'
dtype = 'bfloat16'
compile = False

# Initialize from scratch (no pretrained weights)
init_from = 'scratch'
