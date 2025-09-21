#!/usr/bin/env python3
"""
Prepare tax-law data for instruction tuning
Converts Korean tax law JSON files into instruction-response format
"""

import os
import json
import pickle
import tiktoken
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_tax_law_json(file_path: str) -> Dict[str, Any]:
    """Load and parse a tax law JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_law_articles(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract law articles from JSON data"""
    articles = []
    
    if '법령' not in data:
        return articles
    
    law_data = data['법령']
    
    # Extract basic information
    basic_info = law_data.get('기본정보', {})
    law_name = basic_info.get('법령명_한글', '')
    
    # Extract articles from 조문 section
    if '조문' in law_data:
        # Handle different JSON structures
        if isinstance(law_data['조문'], dict) and '조문단위' in law_data['조문']:
            # Structure: {"조문": {"조문단위": [...]}}
            articles_data = law_data['조문']['조문단위']
        elif isinstance(law_data['조문'], list):
            # Structure: {"조문": [...]}
            articles_data = law_data['조문']
        else:
            return articles
        
        for article in articles_data:
            article_num = article.get('조문번호', '')
            article_title = article.get('조문제목', '')
            article_content = article.get('조문내용', '')
            
            # Skip empty or chapter headers
            if not article_content or article_content.strip() == '':
                continue
            
            # Extract detailed content from 항 (paragraphs)
            full_content = article_content
            if '항' in article and article['항']:
                for paragraph in article['항']:
                    para_num = paragraph.get('항번호', '')
                    para_content = paragraph.get('항내용', '')
                    if para_content and para_content.strip():
                        if para_num:
                            full_content += f"\n{para_num} {para_content}"
                        else:
                            full_content += f"\n{para_content}"
                    
                    # Extract 호 (sub-paragraphs)
                    if '호' in paragraph and paragraph['호']:
                        for sub_para in paragraph['호']:
                            sub_num = sub_para.get('호번호', '')
                            sub_content = sub_para.get('호내용', '')
                            if sub_content and sub_content.strip():
                                if sub_num:
                                    full_content += f"\n{sub_num} {sub_content}"
                                else:
                                    full_content += f"\n{sub_content}"
            
            if full_content.strip() and len(full_content.strip()) > 10:  # Filter out very short content
                articles.append({
                    'law_name': law_name,
                    'article_num': article_num,
                    'article_title': article_title,
                    'content': full_content.strip()
                })
    
    return articles


def create_instruction_pairs(articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Create instruction-response pairs from law articles"""
    instruction_pairs = []
    
    for article in articles:
        law_name = article['law_name']
        article_num = article['article_num']
        article_title = article['article_title']
        content = article['content']
        
        # Create different types of instruction-response pairs
        
        # 1. Article explanation
        if article_title:
            instruction = f"{law_name} 제{article_num}조 ({article_title})에 대해 설명해주세요."
            response = f"{law_name} 제{article_num}조 ({article_title})의 내용은 다음과 같습니다:\n\n{content}"
            instruction_pairs.append({
                'instruction': instruction,
                'response': response
            })
        
        # 2. What is this article about?
        instruction = f"{law_name} 제{article_num}조는 무엇에 관한 조문인가요?"
        response = f"{law_name} 제{article_num}조는 '{article_title}'에 관한 조문입니다.\n\n{content}"
        instruction_pairs.append({
            'instruction': instruction,
            'response': response
        })
        
        # 3. Specific content question
        if len(content) > 100:  # Only for longer articles
            instruction = f"{law_name} 제{article_num}조의 주요 내용을 요약해주세요."
            response = f"{law_name} 제{article_num}조 ({article_title})의 주요 내용:\n\n{content}"
            instruction_pairs.append({
                'instruction': instruction,
                'response': response
            })
        
        # 4. Legal interpretation
        if '세율' in content or '과세' in content or '면세' in content:
            instruction = f"{law_name} 제{article_num}조에서 규정하는 세율이나 과세 기준은 무엇인가요?"
            response = f"{law_name} 제{article_num}조 ({article_title})에서 규정하는 내용:\n\n{content}"
            instruction_pairs.append({
                'instruction': instruction,
                'response': response
            })
    
    return instruction_pairs


def process_all_tax_laws(data_dir: str) -> List[Dict[str, str]]:
    """Process all tax law JSON files in the directory"""
    all_instruction_pairs = []
    
    # Get all subdirectories (tax categories)
    tax_categories = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    print(f"Found {len(tax_categories)} tax categories: {tax_categories}")
    
    for category in tax_categories:
        category_path = os.path.join(data_dir, category)
        print(f"Processing category: {category}")
        
        # Get all JSON files in the category
        json_files = [f for f in os.listdir(category_path) if f.endswith('.json')]
        
        for json_file in json_files:
            file_path = os.path.join(category_path, json_file)
            print(f"  Processing file: {json_file}")
            
            try:
                # Load and process the JSON file
                data = load_tax_law_json(file_path)
                articles = extract_law_articles(data)
                
                # Create instruction pairs
                instruction_pairs = create_instruction_pairs(articles)
                all_instruction_pairs.extend(instruction_pairs)
                
                print(f"    Extracted {len(articles)} articles, created {len(instruction_pairs)} instruction pairs")
                
            except Exception as e:
                print(f"    Error processing {json_file}: {e}")
                continue
    
    return all_instruction_pairs


def save_instruction_data(instruction_pairs: List[Dict[str, str]], output_dir: str):
    """Save instruction data in the format expected by nanoGPT"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to text format
    text_data = []
    for pair in instruction_pairs:
        # Format as instruction-response pair
        text = f"### Instruction:\n{pair['instruction']}\n\n### Response:\n{pair['response']}\n\n"
        text_data.append(text)
    
    # Join all text
    full_text = "".join(text_data)
    
    # Save as text file
    text_file = os.path.join(output_dir, 'input.txt')
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Saved {len(instruction_pairs)} instruction pairs to {text_file}")
    print(f"Total text length: {len(full_text):,} characters")
    
    # Tokenize and save as binary
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(full_text)
    
    # Split into train/val (90/10 split)
    n = len(tokens)
    train_data = tokens[:int(n * 0.9)]
    val_data = tokens[int(n * 0.9):]
    
    # Save binary files
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')
    
    # Convert tokens to numpy array and save as binary
    import numpy as np
    
    train_array = np.array(train_data, dtype=np.uint16)
    val_array = np.array(val_data, dtype=np.uint16)
    
    train_array.tofile(train_file)
    val_array.tofile(val_file)
    
    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'total_tokens': n,
        'train_tokens': len(train_data),
        'val_tokens': len(val_data),
        'instruction_pairs': len(instruction_pairs)
    }
    
    meta_file = os.path.join(output_dir, 'meta.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Saved training data:")
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens: {len(val_data):,}")
    print(f"  Vocab size: {enc.n_vocab:,}")
    print(f"  Instruction pairs: {len(instruction_pairs):,}")


def main():
    """Main function to prepare tax law data"""
    # Set paths
    data_dir = "/Users/prismdata/Documents/nanoGPT/training-data/tax-law"
    output_dir = "/Users/prismdata/Documents/nanoGPT/training-data/tax-law-instruction"
    
    print("🧾 Preparing tax law data for instruction tuning...")
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process all tax law files
    instruction_pairs = process_all_tax_laws(data_dir)
    
    if not instruction_pairs:
        print("❌ No instruction pairs created. Check your data files.")
        return
    
    # Save the processed data
    save_instruction_data(instruction_pairs, output_dir)
    
    print("✅ Tax law instruction data preparation completed!")
    print(f"Created {len(instruction_pairs)} instruction-response pairs")
    
    # Show some examples
    print("\n📋 Sample instruction pairs:")
    for i, pair in enumerate(instruction_pairs[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {pair['instruction']}")
        print(f"Response: {pair['response'][:200]}...")


if __name__ == "__main__":
    main()
