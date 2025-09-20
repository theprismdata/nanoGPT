#!/usr/bin/env python3
"""
Create nanoGPT models locally without downloading pretrained weights.
This demonstrates how nanoGPT can work completely offline.
"""

import torch
import os
from model import GPTConfig, GPT

def create_tiny_model():
    """Create a very small model for quick testing"""
    print("🔧 Creating Tiny GPT Model...")
    config = GPTConfig(
        block_size=128,      # Small context window
        vocab_size=65,       # Shakespeare character vocab
        n_layer=2,           # Just 2 layers
        n_head=2,            # 2 attention heads
        n_embd=64,           # Small embedding dimension
        dropout=0.1,
        bias=False           # No bias for efficiency
    )
    
    model = GPT(config)
    print(f"✅ Tiny model created: {model.get_num_params():,} parameters")
    return model, config

def create_small_model():
    """Create a small but capable model"""
    print("🔧 Creating Small GPT Model...")
    config = GPTConfig(
        block_size=256,      # Reasonable context
        vocab_size=65,       # Shakespeare character vocab
        n_layer=4,           # 4 layers
        n_head=4,            # 4 attention heads
        n_embd=128,          # Medium embedding dimension
        dropout=0.1,
        bias=False
    )
    
    model = GPT(config)
    print(f"✅ Small model created: {model.get_num_params():,} parameters")
    return model, config

def create_medium_model():
    """Create a medium-sized model"""
    print("🔧 Creating Medium GPT Model...")
    config = GPTConfig(
        block_size=512,      # Larger context
        vocab_size=50304,    # GPT-2 vocab size (rounded up for efficiency)
        n_layer=8,           # 8 layers
        n_head=8,            # 8 attention heads
        n_embd=512,          # Larger embedding dimension
        dropout=0.1,
        bias=False
    )
    
    model = GPT(config)
    print(f"✅ Medium model created: {model.get_num_params():,} parameters")
    return model, config

def create_custom_model(layers=6, heads=6, embd=384, context=256, vocab=65):
    """Create a custom model with specified parameters"""
    print(f"🔧 Creating Custom GPT Model...")
    print(f"   Layers: {layers}, Heads: {heads}, Embedding: {embd}")
    print(f"   Context: {context}, Vocab: {vocab}")
    
    config = GPTConfig(
        block_size=context,
        vocab_size=vocab,
        n_layer=layers,
        n_head=heads,
        n_embd=embd,
        dropout=0.1,
        bias=False
    )
    
    model = GPT(config)
    print(f"✅ Custom model created: {model.get_num_params():,} parameters")
    return model, config

def save_model(model, config, name):
    """Save model and config to disk"""
    save_dir = f"local_models/{name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(save_dir, "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_args': {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'block_size': config.block_size,
            'bias': config.bias,
            'vocab_size': config.vocab_size,
            'dropout': config.dropout
        }
    }, model_path)
    
    print(f"💾 Model saved to: {save_dir}/")
    return save_dir

def test_model_generation(model, config):
    """Test the model with a simple forward pass"""
    print("🧪 Testing model generation...")
    
    model.eval()
    with torch.no_grad():
        # Create random input
        batch_size = 1
        seq_len = min(32, config.block_size)
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits, _ = model(x)
        print(f"✅ Input shape: {x.shape}")
        print(f"✅ Output shape: {logits.shape}")
        print(f"✅ Model works correctly!")

def main():
    print("🚀 nanoGPT Local Model Generator")
    print("=" * 50)
    
    models_created = []
    
    # Create different sized models
    print("\n1️⃣ Creating Tiny Model (for testing)")
    tiny_model, tiny_config = create_tiny_model()
    test_model_generation(tiny_model, tiny_config)
    tiny_dir = save_model(tiny_model, tiny_config, "tiny")
    models_created.append(("Tiny", tiny_dir, tiny_model.get_num_params()))
    
    print("\n2️⃣ Creating Small Model (for quick training)")
    small_model, small_config = create_small_model()
    test_model_generation(small_model, small_config)
    small_dir = save_model(small_model, small_config, "small")
    models_created.append(("Small", small_dir, small_model.get_num_params()))
    
    print("\n3️⃣ Creating Medium Model (for better performance)")
    medium_model, medium_config = create_medium_model()
    test_model_generation(medium_model, medium_config)
    medium_dir = save_model(medium_model, medium_config, "medium")
    models_created.append(("Medium", medium_dir, medium_model.get_num_params()))
    
    print("\n4️⃣ Creating Custom Model (Shakespeare optimized)")
    custom_model, custom_config = create_custom_model(
        layers=6, heads=6, embd=384, context=256, vocab=65
    )
    test_model_generation(custom_model, custom_config)
    custom_dir = save_model(custom_model, custom_config, "shakespeare")
    models_created.append(("Shakespeare", custom_dir, custom_model.get_num_params()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Summary of Created Models:")
    print("=" * 50)
    
    for name, path, params in models_created:
        print(f"🤖 {name:12} | {params:>8,} params | {path}")
    
    print("\n💡 Usage Examples:")
    print("# Load a model:")
    print("checkpoint = torch.load('local_models/tiny/model.pt')")
    print("config = GPTConfig(**checkpoint['model_args'])")
    print("model = GPT(config)")
    print("model.load_state_dict(checkpoint['model_state_dict'])")
    
    print("\n# Use in training:")
    print("python train.py config/train_shakespeare_char.py --init_from=resume --out_dir=local_models/tiny")
    
    print(f"\n✅ All models created successfully!")
    print(f"📁 Models saved in: local_models/")

if __name__ == "__main__":
    main()
