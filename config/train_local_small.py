# Configuration for training with locally generated small model
# This config uses a small model created entirely locally

out_dir = 'out-local-small'
eval_interval = 200
eval_iters = 50
log_interval = 10

# Save checkpoints when validation improves
always_save_checkpoint = False

# No wandb logging for local testing
wandb_log = False

# Dataset configuration
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256  # Match our local model's context length

# Model architecture - matches our locally created small model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

# Learning rate and training schedule
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99

# Warmup
warmup_iters = 100

# For Mac M4 optimization
# device = 'mps'
# compile = False

# Use locally created model as starting point
init_from = 'scratch'  # We'll override this to use our local model
