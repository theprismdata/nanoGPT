"""
Prepare instruction-following dataset for nanoGPT training.
This script creates instruction-response pairs in a format suitable for nanoGPT.
"""
import os
import pickle
import numpy as np
import tiktoken
import json

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(__file__), exist_ok=True)

# Example instruction dataset - you can replace this with your own data
instruction_data = [
    {
        "instruction": "Translate the following English sentence to Korean:",
        "input": "Hello, how are you?",
        "output": "안녕하세요, 어떻게 지내세요?"
    },
    {
        "instruction": "Write a short poem about nature.",
        "input": "",
        "output": "The gentle breeze whispers through the trees,\nBirds sing their songs with graceful ease,\nFlowers bloom in colors bright,\nNature's beauty, pure delight."
    },
    {
        "instruction": "Explain what machine learning is in simple terms.",
        "input": "",
        "output": "Machine learning is a way for computers to learn patterns from data without being explicitly programmed. It's like teaching a computer to recognize pictures of cats by showing it thousands of cat photos, so it can identify cats in new photos it hasn't seen before."
    },
    {
        "instruction": "Summarize the following text:",
        "input": "Artificial intelligence has revolutionized many industries. From healthcare to finance, AI systems are helping professionals make better decisions, automate routine tasks, and discover new insights from large amounts of data.",
        "output": "AI has transformed various industries by helping professionals make better decisions, automate tasks, and find insights from big data."
    },
    {
        "instruction": "Write a function in Python that calculates the factorial of a number.",
        "input": "",
        "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"
    },
    # Add more examples...
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris."
    },
    {
        "instruction": "Convert the following temperature from Celsius to Fahrenheit:",
        "input": "25°C",
        "output": "25°C is equal to 77°F. The formula is: F = (C × 9/5) + 32"
    },
    {
        "instruction": "Write a professional email response to decline a meeting invitation.",
        "input": "",
        "output": "Subject: Re: Meeting Invitation\n\nDear [Name],\n\nThank you for the meeting invitation. Unfortunately, I have a scheduling conflict and won't be able to attend. I would appreciate it if you could share the meeting notes or key decisions afterward.\n\nPlease let me know if we can reschedule for another time that works for everyone.\n\nBest regards,\n[Your Name]"
    },
    {
        "instruction": "Explain the difference between supervised and unsupervised learning.",
        "input": "",
        "output": "Supervised learning uses labeled data to train models, like teaching a computer to recognize cats using photos labeled 'cat' or 'not cat'. Unsupervised learning finds patterns in unlabeled data, like grouping customers by purchasing behavior without knowing the groups beforehand."
    },
    {
        "instruction": "Create a simple recipe for chocolate chip cookies.",
        "input": "",
        "output": "Ingredients: 2¼ cups flour, 1 tsp baking soda, 1 cup butter, ¾ cup sugar, ¾ cup brown sugar, 2 eggs, 2 tsp vanilla, 2 cups chocolate chips.\n\nInstructions: 1) Preheat oven to 375°F. 2) Mix dry ingredients. 3) Cream butter and sugars, add eggs and vanilla. 4) Combine wet and dry ingredients, fold in chocolate chips. 5) Drop spoonfuls on baking sheet. 6) Bake 9-11 minutes until golden brown."
    }
]

def format_instruction_data(data_list):
    """
    Format instruction data into a training format.
    Uses special tokens to separate instruction, input, and output.
    """
    formatted_text = ""
    
    for item in data_list:
        # Create a structured format for instruction tuning
        text = f"<|instruction|>{item['instruction']}"
        
        if item['input'].strip():
            text += f"<|input|>{item['input']}"
        
        text += f"<|response|>{item['output']}<|end|>\n\n"
        
        formatted_text += text
    
    return formatted_text

def create_extended_dataset(base_data, multiplier=50):
    """
    Create a larger dataset by duplicating and slightly varying the base examples.
    This is just for demonstration - in practice, you'd want real diverse data.
    """
    extended_data = []
    
    for _ in range(multiplier):
        extended_data.extend(base_data)
    
    return extended_data

if __name__ == '__main__':
    print("Preparing instruction dataset...")
    
    # Create extended dataset for better training
    extended_data = create_extended_dataset(instruction_data, multiplier=100)
    
    # Format the data
    formatted_text = format_instruction_data(extended_data)
    
    print(f"Total dataset length: {len(formatted_text):,} characters")
    print(f"Number of instruction examples: {len(extended_data):,}")
    
    # Split into train/val
    n = len(formatted_text)
    train_data = formatted_text[:int(n*0.9)]
    val_data = formatted_text[int(n*0.9):]
    
    # Use GPT-2 BPE tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Add special tokens to the encoding (in practice, you might want to properly extend the tokenizer)
    train_ids = enc.encode(train_data)
    val_ids = enc.encode(val_data)
    
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    
    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    
    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'special_tokens': ['<|instruction|>', '<|input|>', '<|response|>', '<|end|>'],
        'dataset_info': {
            'total_examples': len(extended_data),
            'train_tokens': len(train_ids),
            'val_tokens': len(val_ids)
        }
    }
    
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("Dataset preparation complete!")
    print(f"Files saved to: {os.path.dirname(__file__)}")
    
    # Show a sample of the formatted data
    print("\n" + "="*50)
    print("Sample of formatted data:")
    print("="*50)
    print(formatted_text[:500] + "...")
