#!/usr/bin/env python3
"""
Foundation Model Configuration Generator

Creates optimized training configurations for different sizes of foundation models.
Usage: python create_foundation_config.py --size small
"""

import argparse
import os

def create_foundation_config(size="small", stages=False):
    """Create foundation model training configurations"""
    
    configs = {
        "tiny": {
            "n_layer": 6, "n_head": 6, "n_embd": 384,
            "block_size": 512, "batch_size": 16,
            "max_iters": 50000, "learning_rate": 1e-3,
            "params_estimate": "25M"
        },
        "small": {
            "n_layer": 12, "n_head": 12, "n_embd": 768,
            "block_size": 1024, "batch_size": 12,
            "max_iters": 200000, "learning_rate": 6e-4,
            "params_estimate": "124M"
        },
        "medium": {
            "n_layer": 24, "n_head": 16, "n_embd": 1024,
            "block_size": 2048, "batch_size": 8,
            "max_iters": 400000, "learning_rate": 4e-4,
            "params_estimate": "350M"
        },
        "large": {
            "n_layer": 36, "n_head": 20, "n_embd": 1280,
            "block_size": 4096, "batch_size": 4,
            "max_iters": 600000, "learning_rate": 3e-4,
            "params_estimate": "774M"
        }
    }
    
    if size not in configs:
        raise ValueError(f"Size must be one of: {list(configs.keys())}")
    
    config = configs[size]
    
    # Base configuration template
    base_config = f'''# Foundation Model Training Configuration - {size.upper()}
# Estimated parameters: {config["params_estimate"]}
# Training from scratch without any pretrained weights

# Output directory
out_dir = 'out-foundation-{size}'

# Evaluation settings
eval_interval = 2000
eval_iters = 200
log_interval = 100

# Checkpoint settings
always_save_checkpoint = True

# Logging (optional)
wandb_log = False  # Set to True if you want to track experiments
wandb_project = 'foundation-model'
wandb_run_name = 'foundation-{size}-v1'

# Data configuration
dataset = 'foundation'  # Prepare your foundation dataset
gradient_accumulation_steps = 8  # Simulate larger batch sizes
batch_size = {config["batch_size"]}
block_size = {config["block_size"]}

# Model architecture - {size.upper()} Foundation Model
n_layer = {config["n_layer"]}
n_head = {config["n_head"]}
n_embd = {config["n_embd"]}
dropout = 0.0  # No dropout for pretraining
bias = False   # Remove bias for efficiency

# Optimizer settings (based on GPT-3 paper)
learning_rate = {config["learning_rate"]}
max_iters = {config["max_iters"]}
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = {config["max_iters"]}
min_lr = {config["learning_rate"] / 10}

# System settings (Mac M4 optimized)
device = 'mps'
dtype = 'bfloat16'
compile = False

# Initialize from scratch (no pretrained weights)
init_from = 'scratch'
'''

    # Write base config
    config_path = f'config/train_foundation_{size}.py'
    with open(config_path, 'w') as f:
        f.write(base_config)
    
    print(f"✅ Created foundation config: {config_path}")
    
    if stages:
        create_staged_configs(size, config)
    
    return config_path

def create_staged_configs(size, base_config):
    """Create multi-stage training configurations"""
    
    stages = [
        {
            "name": "stage1_foundation",
            "description": "Initial foundation training",
            "max_iters": base_config["max_iters"] // 2,
            "learning_rate": base_config["learning_rate"],
            "init_from": "scratch"
        },
        {
            "name": "stage2_stabilization", 
            "description": "Stabilization with lower learning rate",
            "max_iters": base_config["max_iters"] // 3,
            "learning_rate": base_config["learning_rate"] / 2,
            "init_from": "resume"
        },
        {
            "name": "stage3_refinement",
            "description": "Final refinement",
            "max_iters": base_config["max_iters"] // 6,
            "learning_rate": base_config["learning_rate"] / 4,
            "init_from": "resume"
        }
    ]
    
    for stage in stages:
        stage_config = f'''# Foundation Model Training - {size.upper()} - {stage["name"].upper()}
# {stage["description"]}

# Import base config
exec(open('config/train_foundation_{size}.py').read())

# Override stage-specific settings
out_dir = 'out-foundation-{size}-{stage["name"]}'
max_iters = {stage["max_iters"]}
lr_decay_iters = {stage["max_iters"]}
learning_rate = {stage["learning_rate"]}
min_lr = {stage["learning_rate"] / 10}
init_from = '{stage["init_from"]}'
wandb_run_name = 'foundation-{size}-{stage["name"]}'

print("🚀 Stage: {stage["name"]} - {stage["description"]}")
'''
        
        stage_path = f'config/train_foundation_{size}_{stage["name"]}.py'
        with open(stage_path, 'w') as f:
            f.write(stage_config)
        
        print(f"✅ Created stage config: {stage_path}")

def create_data_preparation_script():
    """Create foundation data preparation script"""
    
    script_content = '''#!/usr/bin/env python3
"""
Foundation Dataset Preparation Script

Prepares large-scale text data for foundation model training.
Combines multiple data sources and creates train/val splits.
"""

import os
import numpy as np
import tiktoken
from pathlib import Path

def prepare_foundation_data():
    """Prepare foundation model dataset"""
    
    print("🔄 Preparing foundation dataset...")
    
    # Create data directory
    data_dir = Path("data/foundation")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Collect all text files
    text_files = []
    
    # Look for text files in common locations
    search_paths = [
        "data/texts/",
        "data/corpus/", 
        "data/raw/",
        "./texts/",
        "./corpus/"
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            text_files.extend(Path(search_path).glob("*.txt"))
    
    if not text_files:
        print("❌ No text files found!")
        print("📁 Please place your text files in one of these directories:")
        for path in search_paths:
            print(f"   - {path}")
        print("\\n💡 Or modify this script to point to your data location")
        return
    
    print(f"📚 Found {len(text_files)} text files")
    
    # Combine all text
    all_text = ""
    total_chars = 0
    
    for text_file in text_files:
        print(f"📖 Processing: {text_file.name}")
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
                all_text += content + "\\n\\n"  # Add separation between files
                total_chars += len(content)
        except Exception as e:
            print(f"⚠️  Warning: Could not read {text_file}: {e}")
    
    if not all_text:
        print("❌ No text content found!")
        return
    
    print(f"📊 Total characters: {total_chars:,}")
    
    # Create train/validation split (95/5)
    split_idx = int(len(all_text) * 0.95)
    train_text = all_text[:split_idx]
    val_text = all_text[split_idx:]
    
    print("🔤 Tokenizing text...")
    
    # Tokenize
    train_ids = enc.encode(train_text)
    val_ids = enc.encode(val_text)
    
    print(f"🔢 Train tokens: {len(train_ids):,}")
    print(f"🔢 Val tokens: {len(val_ids):,}")
    
    # Save as binary files
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    
    np.array(train_ids, dtype=np.uint16).tofile(train_path)
    np.array(val_ids, dtype=np.uint16).tofile(val_path)
    
    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'total_chars': total_chars,
        'num_files': len(text_files)
    }
    
    import pickle
    with open(data_dir / "meta.pkl", 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"✅ Foundation dataset prepared!")
    print(f"📁 Files saved to: {data_dir}")
    print(f"📈 Ready for training with dataset='foundation'")

if __name__ == "__main__":
    prepare_foundation_data()
'''
    
    script_path = "data/foundation/prepare.py"
    os.makedirs("data/foundation", exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"✅ Created data preparation script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Create foundation model configurations")
    parser.add_argument("--size", choices=["tiny", "small", "medium", "large"], 
                       default="small", help="Model size")
    parser.add_argument("--stages", action="store_true", 
                       help="Create multi-stage training configs")
    parser.add_argument("--data-prep", action="store_true",
                       help="Create data preparation script")
    
    args = parser.parse_args()
    
    print("🏗️  Foundation Model Configuration Generator")
    print("=" * 50)
    
    # Create main config
    config_path = create_foundation_config(args.size, args.stages)
    
    # Create data preparation script if requested
    if args.data_prep:
        create_data_preparation_script()
    
    print("\\n" + "=" * 50)
    print("📋 Next Steps:")
    print("=" * 50)
    print("1. 📚 Prepare your foundation dataset:")
    print("   python data/foundation/prepare.py")
    print("\\n2. 🚀 Start foundation training:")
    print(f"   python train.py {config_path} --device=mps --compile=False")
    print("\\n3. 📊 Monitor training progress:")
    print("   tail -f train.log")
    print("\\n4. 🧪 Test your foundation model:")
    print(f"   python sample.py --out_dir=out-foundation-{args.size} --device=mps")
    
    if args.stages:
        print("\\n🔄 For staged training, run each stage sequentially:")
        print(f"   python train.py config/train_foundation_{args.size}_stage1_foundation.py --device=mps --compile=False")
        print(f"   python train.py config/train_foundation_{args.size}_stage2_stabilization.py --device=mps --compile=False") 
        print(f"   python train.py config/train_foundation_{args.size}_stage3_refinement.py --device=mps --compile=False")

if __name__ == "__main__":
    main()
