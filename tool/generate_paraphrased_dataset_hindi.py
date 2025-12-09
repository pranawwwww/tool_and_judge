"""
Generate paraphrased Hindi dataset variants.
Uses OpenAI GPT-4o-mini for paraphrasing while preserving meaning.
"""

import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import os
from parse_dataset import load_json_lines

# Load API keys from .env file
load_dotenv(dotenv_path=".env")

# Initialize OpenAI client for Hindi paraphrasing
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

postfix_to_generate = [
    "_hi_full",
    "_hi_partial"
]

system_prompt = '''You are a helpful assistant helping with rephrasing Hindi user requests while accurately preserving their meaning, including numbers and names if they exist. 

IMPORTANT RULES:
1. Keep all English words and numbers unchanged
2. Do not answer the requirement, just produce another one that is identical in meaning but phrased differently
3. Preserve technical terms and proper nouns
4. Maintain the same overall structure and intent
5. Produce ONLY the rephrased requirement, without further thoughts or explanations

Example:
USER: क्या आप एक त्रिभुज के आयाम और गुण खोज सकते हैं, यदि यह ज्ञात है कि इसकी तीन भुजाएं 5 इकाई, 4 इकाई और 3 इकाई लंबी हैं?

ASSISTANT: यदि किसी त्रिभुज की तीन भुजाओं की लंबाई 5, 4 और 3 इकाइयाँ हैं, तो आप इसके आयाम और गुणों का पता लगा सकते हैं?
'''

def generate_paraphrased_case(question: str) -> str:
    """Generate a paraphrased version of a Hindi question using OpenAI."""
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
        print(f"Error generating paraphrase: {e}")
        return question  # Return original if error


for postfix in postfix_to_generate:
    print(f"Generating paraphrased dataset for postfix: {postfix}")
    original_dataset_path = f'dataset/BFCL_v4_multiple{postfix}.json'
    paraphrased_dataset_path = f'dataset/BFCL_v4_multiple{postfix}_para.json'
    
    # Check if original dataset exists
    if not os.path.exists(original_dataset_path):
        print(f"Original dataset not found at {original_dataset_path}. Skipping...")
        continue
    
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        original_data = load_json_lines(f)
    
    paraphrased_data = []
    existing_indices = []
    
    try:
        with open(paraphrased_dataset_path, 'r', encoding='utf-8') as f:
            paraphrased_data = load_json_lines(f)
            existing_indices = [item['id'] for item in paraphrased_data]
    except FileNotFoundError:
        print(f"No existing paraphrased dataset found at {paraphrased_dataset_path}. A new one will be created.")
    
    with open(paraphrased_dataset_path, 'w', encoding='utf-8') as f:
        warning_printed = False
        processed_count = 0
        for item in original_data:
            id = item['id']
            if id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items in {paraphrased_dataset_path}.")
                    warning_printed = True
                paraphrased_data.append(item)
                continue
            
            try:
                question_content = item['question'][0][0]['content']
                print(f"  Processing {id}... ", end="", flush=True)
                
                paraphrased_question = generate_paraphrased_case(question_content)
                
                paraphrased_item = item.copy()
                paraphrased_item['question'][0][0]['content'] = paraphrased_question
                paraphrased_data.append(paraphrased_item)
                processed_count += 1
                print(f"✓")
            except Exception as e:
                print(f"✗ (Error: {str(e)[:50]})")
                paraphrased_data.append(item)  # Keep original if error
        
        # Sort by ID
        paraphrased_data = sorted(
            paraphrased_data,
            key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
        )
        
        # Write all data
        f.seek(0)
        f.truncate()
        for n in paraphrased_data:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
        f.flush()
    
    print(f"Paraphrased dataset with {len(paraphrased_data)} items saved to {paraphrased_dataset_path}.")
    print(f"  (Newly processed: {processed_count} items)\n")
