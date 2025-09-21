#!/usr/bin/env python3
"""
Instruction Model Testing Script for nanoGPT

This script provides an easy way to test the instruction-tuned model
with various types of questions and prompts.

Usage:
    python test_instruction.py --question "What is machine learning?"
    python test_instruction.py --interactive
    python test_instruction.py --batch
"""

import argparse
import subprocess
import sys
import os

def run_sample(instruction, input_text="", temperature=0.7, max_tokens=200, num_samples=1):
    """Run the sample.py script with instruction format"""
    
    # Format the prompt according to instruction format
    if input_text.strip():
        prompt = f"<|instruction|>{instruction}<|input|>{input_text}<|response|>"
    else:
        prompt = f"<|instruction|>{instruction}<|response|>"
    
    # Build the command
    cmd = [
        "python", "sample.py",
        "--out_dir=out-instruction",
        "--device=mps",
        f"--start={prompt}",
        f"--num_samples={num_samples}",
        f"--max_new_tokens={max_tokens}",
        f"--temperature={temperature}"
    ]
    
    print(f"🤖 Question: {instruction}")
    if input_text.strip():
        print(f"📝 Input: {input_text}")
    print(f"🔄 Generating response...")
    print("-" * 50)
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            # Parse the output to extract just the response
            output = result.stdout
            lines = output.split('\n')
                        # Find the line that contains our prompt and extract the response
            response = None
            for line in lines:
                if prompt in line:
                    # Extract everything after the prompt
                    response_part = line.split(prompt, 1)[-1]
                    if response_part:
                        # Clean up the response
                        response = response_part.split('<|end|>')[0].strip()
                        # Remove any additional instruction patterns that might follow
                        if '<|instruction|>' in response:
                            response = response.split('<|instruction|>')[0].strip()
                        break
            
            if response:
                print(f"💡 Response: {response}")
            else:
                print("❌ Could not parse response from output")
                print("Raw output lines containing our prompt:")
                for line in lines:
                    if prompt in line:
                        print(f"  -> {line}")
                if not any(prompt in line for line in lines):
                    print("❌ Prompt not found in output. Full output:")
                    print(output)
        else:
            print(f"❌ Error running model: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

def interactive_mode():
    """Interactive mode for testing"""
    print("🎯 Interactive Instruction Testing Mode")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50)
    
    while True:
        try:
            instruction = input("\n📝 Enter your instruction: ").strip()
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not instruction:
                continue
                
            # Ask for optional input
            input_text = input("📄 Enter input (optional, press Enter to skip): ").strip()
            
            # Ask for temperature
            temp_input = input("🌡️  Temperature (0.1-1.0, default 0.7): ").strip()
            try:
                temperature = float(temp_input) if temp_input else 0.7
                temperature = max(0.1, min(1.0, temperature))  # Clamp between 0.1 and 1.0
            except:
                temperature = 0.7
            
            run_sample(instruction, input_text, temperature)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_test():
    """Run a batch of predefined tests"""
    print("🔄 Running batch tests...")
    print("=" * 50)
    
    test_cases = [
        {
            "instruction": "What is the capital of South Korea?",
            "input": "",
            "temperature": 0.5
        },
        {
            "instruction": "Write a short poem about technology.",
            "input": "",
            "temperature": 0.8
        },
        {
            "instruction": "Explain quantum computing in simple terms.",
            "input": "",
            "temperature": 0.6
        },
        {
            "instruction": "Translate the following English sentence to Korean:",
            "input": "I love programming with Python.",
            "temperature": 0.5
        },
        {
            "instruction": "Write a Python function to reverse a string.",
            "input": "",
            "temperature": 0.4
        },
        {
            "instruction": "Summarize the following text:",
            "input": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.",
            "temperature": 0.6
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}/{len(test_cases)}")
        run_sample(test["instruction"], test["input"], test["temperature"])
        
        if i < len(test_cases):
            input("\nPress Enter to continue to next test...")

def main():
    parser = argparse.ArgumentParser(description="Test nanoGPT instruction model")
    parser.add_argument("--question", "-q", type=str, help="Single question to ask")
    parser.add_argument("--input", "-i", type=str, default="", help="Input text for the question")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature (0.1-1.0)")
    parser.add_argument("--max_tokens", "-m", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", action="store_true", help="Run batch tests")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists("out-instruction/ckpt.pt"):
        print("❌ Error: No instruction model found!")
        print("Please train the instruction model first:")
        print("python train.py config/train_instruction.py --device=mps --compile=False")
        sys.exit(1)
    
    if args.interactive:
        interactive_mode()
    elif args.batch:
        batch_test()
    elif args.question:
        run_sample(args.question, args.input, args.temperature, args.max_tokens)
    else:
        print("Please specify --question, --interactive, or --batch mode")
        print("Use --help for more information")

if __name__ == "__main__":
    main()
