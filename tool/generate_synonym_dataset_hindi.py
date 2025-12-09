"""
Generate synonym-based Hindi dataset variants.
Uses OpenAI GPT-4o-mini to replace Hindi words with synonyms while preserving English terms.
"""

import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import os
from parse_dataset import load_json_lines

# Load API keys from .env file
load_dotenv(dotenv_path=".env")

# Initialize OpenAI client for Hindi synonym replacement
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

postfix_to_generate = [
    "_hi_full",
    "_hi_partial"
]

system_prompt = '''You are a helpful assistant that replaces Hindi words with appropriate synonyms while maintaining semantic correctness.

IMPORTANT RULES:
1. Replace ONLY Hindi words with appropriate synonyms of similar meaning
2. KEEP all English words and numbers unchanged
3. Maintain the semantic meaning and grammatical structure
4. Do NOT perform general paraphrasing, only synonym replacement
5. Process thoughtfully - replace words that have good synonyms, keep others if no good synonym exists
6. Do not change technical terms, proper nouns, or words that are part of the question's core meaning
7. Produce ONLY the modified text with synonyms, without further thoughts or explanations

Example:
USER: क्या आप एक स्वस्थ दोपहर के भोजन के व्यंजन को खोज सकते हैं जो 500 कैलोरी से कम है और जिसमें chicken और mushrooms हो?

ASSISTANT: क्या आप एक पौष्टिक मध्याह्न भोजन की विधि को ढूंढ सकते हैं जो 500 कैलोरी से कम है और जिसमें chicken और mushrooms शामिल हो?

(Note: स्वस्थ → पौष्टिक, दोपहर के भोजन → मध्याह्न भोजन, व्यंजन → विधि, खोज → ढूंढ, but chicken and mushrooms remain unchanged)
'''

def generate_synonym_case(question: str) -> str:
    """Generate a synonym-replaced version of a Hindi question using OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating synonyms: {e}")
        return question  # Return original if error


for postfix in postfix_to_generate:
    print(f"Generating synonym dataset for postfix: {postfix}")
    original_dataset_path = f'dataset/BFCL_v4_multiple{postfix}.json'
    synonym_dataset_path = f'dataset/BFCL_v4_multiple{postfix}_syno.json'
    
    # Check if original dataset exists
    if not os.path.exists(original_dataset_path):
        print(f"Original dataset not found at {original_dataset_path}. Skipping...")
        continue
    
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        original_data = load_json_lines(f)
    
    synonym_data = []
    existing_indices = []
    
    try:
        with open(synonym_dataset_path, 'r', encoding='utf-8') as f:
            synonym_data = load_json_lines(f)
            existing_indices = [item['id'] for item in synonym_data]
    except FileNotFoundError:
        print(f"No existing synonym dataset found at {synonym_dataset_path}. A new one will be created.")
    
    with open(synonym_dataset_path, 'w', encoding='utf-8') as f:
        warning_printed = False
        processed_count = 0
        for item in original_data:
            id = item['id']
            if id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items in {synonym_dataset_path}.")
                    warning_printed = True
                synonym_data.append(item)
                continue
            
            try:
                question_content = item['question'][0][0]['content']
                print(f"  Processing {id}... ", end="", flush=True)
                
                synonym_question = generate_synonym_case(question_content)
                
                synonym_item = item.copy()
                synonym_item['question'][0][0]['content'] = synonym_question
                synonym_data.append(synonym_item)
                processed_count += 1
                print(f"✓")
            except Exception as e:
                print(f"✗ (Error: {str(e)[:50]})")
                synonym_data.append(item)  # Keep original if error
        
        # Sort by ID
        synonym_data = sorted(
            synonym_data,
            key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
        )
        
        # Write all data
        f.seek(0)
        f.truncate()
        for n in synonym_data:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
        f.flush()
    
    print(f"Synonym dataset with {len(synonym_data)} items saved to {synonym_dataset_path}.")
    print(f"  (Newly processed: {processed_count} items)\n")
