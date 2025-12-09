"""
Fully translate BFCL questions to Hindi.
Uses OpenAI GPT-4o-mini for translation.
"""

import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from parse_dataset import load_json_lines

# Load API keys from .env file
load_dotenv(dotenv_path=".env")

# Initialize OpenAI client for Hindi translation
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)

system_prompt_full = """You are a professional translation assistant. Your task is to translate the user's question into Hindi accurately and naturally. Do not answer the question. Do not add any line breaks or extra formatting. Provide only the translated Hindi text."""

system_prompt_partial = """You are a translation assistant. The user will provide a question and a JSON string containing important terms.

If any of those important terms appear in the question, keep them unchanged (do not translate them), and translate all other parts into Hindi.

Rules:
1. Preserve all English terms that appear in the JSON string exactly as they are
2. Preserve all numbers exactly as they are
3. Translate only the non-English parts to Hindi naturally
4. Do not answer the question, just translate it
5. Do not add line breaks or extra formatting

Provide only the translated Hindi text."""


def translate_question_openai(question: str, system_prompt: str, important_terms: str = None) -> str:
    """Translate a question to Hindi using OpenAI."""
    try:
        if important_terms:
            user_content = f"{question}\n{important_terms}"
        else:
            user_content = question
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error translating with OpenAI: {e}")
        return None


# Configuration for full and partial translations
translate_configs = [
    {"option": "FULLY_TRANSLATED", "postfix": "_hi_full"},
    {"option": "PARTIALLY_TRANSLATED", "postfix": "_hi_partial"},
]

# Load ground truth for partial translation
with open("dataset/possible_answer/BFCL_v4_multiple.json", "r", encoding="utf-8") as f:
    possible_answers = load_json_lines(f)
    possible_answers_map = {ans["id"]: ans for ans in possible_answers}

for config in translate_configs:
    option = config["option"]
    postfix = config["postfix"]
    
    print(f"\n{'='*60}")
    print(f"Generating {option} Hindi dataset")
    print(f"{'='*60}")
    
    input_path = "dataset/BFCL_v4_multiple.json"
    output_path = f"dataset/BFCL_v4_multiple{postfix}.json"
    
    # Load input dataset
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = load_json_lines(f)
    
    # Load existing translations
    translated_lines = []
    existing_indices = []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            translated_lines = load_json_lines(f)
            existing_indices = [item["id"] for item in translated_lines]
        print(f"Found {len(existing_indices)} existing translations")
    except FileNotFoundError:
        print(f"Creating new translation file")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        processed_count = 0
        
        for i, item in enumerate(dataset):
            item_id = item["id"]
            
            # Skip already processed
            if item_id in existing_indices:
                translated_lines.append(item)
                continue
            
            question = item["question"][0][0]["content"]
            
            print(f"[{i+1}/{len(dataset)}] {item_id}: ", end="", flush=True)
            
            try:
                if option == "FULLY_TRANSLATED":
                    translated = translate_question_openai(
                        question,
                        system_prompt_full
                    )
                else:  # PARTIALLY_TRANSLATED
                    # Get important terms from ground truth
                    possible_answer = possible_answers_map.get(item_id)
                    if not possible_answer:
                        print("⚠ (no ground truth, skipping)")
                        translated_lines.append(item)
                        continue
                    
                    ground_truth_first = possible_answer['ground_truth'][0]
                    important_terms = next(iter(ground_truth_first.values()))
                    important_terms_json = json.dumps(important_terms, ensure_ascii=False)
                    
                    translated = translate_question_openai(
                        question,
                        system_prompt_partial,
                        important_terms_json
                    )
                
                if translated:
                    translated_item = item.copy()
                    translated_item["question"] = [[{
                        "role": "user",
                        "content": translated
                    }]]
                    translated_lines.append(translated_item)
                    processed_count += 1
                    print("✓")
                else:
                    print("✗ (translation failed)")
                    translated_lines.append(item)
                
            except Exception as e:
                print(f"✗ ({str(e)[:40]})")
                translated_lines.append(item)
        
        # Sort by ID and save
        sorted_lines = sorted(
            translated_lines,
            key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
        )
        
        f_out.seek(0)
        f_out.truncate()
        for line in sorted_lines:
            f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
        f_out.flush()
    
    print(f"\n✅ {option} complete!")
    print(f"   Output: {output_path}")
    print(f"   Total items: {len(translated_lines)}")
    print(f"   Newly translated: {processed_count}")
