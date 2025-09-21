#!/usr/bin/env python3
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
    data_dir = Path("training-data/foundation")
    data_dir.mkdir(parents=True, exist_ok=True)
    
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
        print("\n💡 Or modify this script to point to your data location")
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
                all_text += content + "\n\n"  # Add separation between files
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
