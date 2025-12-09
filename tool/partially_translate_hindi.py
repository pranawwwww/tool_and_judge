"""
Partially translate BFCL questions to Hindi while preserving keywords from ground truth.
Uses OpenAI GPT-4o-mini for translation.
"""

import argparse
import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from parse_dataset import load_json_lines

# Load API keys from .env file
load_dotenv(dotenv_path=".env")


def create_translation_prompt(
    question_content: str,
    functions: list[dict],
    ground_truth: list[dict]
) -> str:
    """
    Create a prompt for OpenAI to partially translate to Hindi.
    The model should preserve keywords that appear in ground truth parameter values.
    """
    prompt = f"""## कार्य पृष्ठभूमि

हम एक परीक्षण डेटासेट बना रहे हैं जो AI मॉडल की हिंदी में तर्क करने और फ़ंक्शन कॉल करने की क्षमता का मूल्यांकन करता है। मूल प्रश्न अंग्रेजी में है, और हमें इसे हिंदी में अनुवाद करना है।

**मुख्य चुनौती**: जब प्रश्न को पूरी तरह से हिंदी में अनुवाद किया जाता है, तो मॉडल फ़ंक्शन पैरामीटर को भी हिंदी में भरने का प्रयास करता है (उदाहरण के लिए "New York" को "न्यूयॉर्क" में अनुवाद करना), जिससे अपेक्षित अंग्रेजी ground truth से मेल नहीं खाता। इसलिए, हमें **आंशिक अनुवाद** करना है - प्रश्न के मुख्य भाग को हिंदी में अनुवाद करें, लेकिन वे महत्वपूर्ण जानकारी जो फ़ंक्शन कॉल पैरामीटर में दिखाई देगी, अंग्रेजी में रखें।

## मूल प्रश्न (अंग्रेजी)
{question_content}

## उपलब्ध फ़ंक्शन परिभाषा
{json.dumps(functions, ensure_ascii=False, indent=2)}

## अपेक्षित फ़ंक्शन कॉल (Ground Truth)
{json.dumps(ground_truth, ensure_ascii=False, indent=2)}

## आपका कार्य

उपरोक्त अंग्रेजी प्रश्न को हिंदी में अनुवाद करें, लेकिन सुनिश्चित करें कि अनुवादित प्रश्न AI मॉडल को ground truth से पूरी तरह से मेल खाने वाली फ़ंक्शन कॉल उत्पन्न करने के लिए प्रेरित करता है।

## अनुवाद प्रक्रिया

1. **Ground truth का विश्लेषण करें**: ध्यान से देखें कि ground truth में कौन सी जानकारी फ़ंक्शन नाम और पैरामीटर मान हैं, और पहचानें कि कौन सी जानकारी मॉडल को सही तरीके से निकालने के लिए अंग्रेजी में रहनी चाहिए।

2. **संरक्षण रणनीति निर्धारित करें**: तय करें कि कौन सी शब्दावली, संख्याएं, नाम, तारीखें आदि अंग्रेजी में रहनी चाहिए।

3. **अनुवाद करें**: प्रश्न को प्राकृतिक और प्रवाहमान हिंदी में अनुवाद करें, जबकि आवश्यक महत्वपूर्ण जानकारी को अंग्रेजी में रखें।

4. **सत्यापन करें**: कल्पना करें कि एक AI मॉडल आपके अनुवादित प्रश्न और फ़ंक्शन परिभाषा को देखता है। क्या यह:
   - सही फ़ंक्शन चुनेगा?
   - ground truth से मेल खाने वाले पैरामीटर मान निकाल सकेगा?

5. **यदि आवश्यक हो तो समायोजित करें**: यदि सत्यापन से समस्याएं पता चलती हैं, तो सही पैरामीटर निष्कर्षण सुनिश्चित करने के लिए अनुवाद को संशोधित करें।

## आउटपुट आवश्यकता

केवल अनुवादित प्रश्न आउटपुट करें, कोई व्याख्या, मूल पाठ, या विचार प्रक्रिया शामिल न करें।"""

    return prompt


def translate_with_openai(
    client: OpenAI,
    prompt: str,
    model: str = "gpt-4o-mini"
) -> tuple[str | None, str]:
    """
    Use OpenAI to translate the question to Hindi.

    Args:
        client: OpenAI client
        prompt: The translation prompt
        model: Model name to use

    Returns:
        tuple: (translated_text, raw_response)
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent translations
            max_tokens=1000
        )

        translated_text = response.choices[0].message.content.strip()
        return translated_text, translated_text

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None, ""


def main():
    # === Parse command line arguments ===
    parser = argparse.ArgumentParser(
        description="Partially translate BFCL questions to Hindi while preserving English keywords"
    )
    parser.add_argument(
        "--input",
        default="dataset/BFCL_v4_multiple.json",
        help="Input dataset file"
    )
    parser.add_argument(
        "--ground-truth",
        default="dataset/possible_answer/BFCL_v4_multiple.json",
        help="Ground truth file"
    )
    parser.add_argument(
        "--output",
        default="dataset/BFCL_v4_multiple_hi_partial.json",
        help="Output file"
    )

    args = parser.parse_args()

    # === Configuration ===
    input_file = args.input
    ground_truth_file = args.ground_truth
    output_file = args.output

    # OpenAI configuration
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env")

    # Initialize the client
    client = OpenAI(api_key=api_key)

    print("Using OpenAI GPT-4o-mini for partial translation to Hindi")

    # === Load input files ===
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = load_json_lines(f)

    print(f"Loading ground truth from {ground_truth_file}...")
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truths = load_json_lines(f)

    # Create a mapping of id to ground truth
    gt_map = {gt['id']: gt['ground_truth'] for gt in ground_truths}

    # === Load existing translations (for resumption) ===
    existing_indices = []
    translated_lines = []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            translated_lines = load_json_lines(f)
            existing_indices = [item["id"] for item in translated_lines]
        print(f"Found {len(existing_indices)} existing translations in {output_file}")
    except FileNotFoundError:
        print(f"No existing translated dataset found at {output_file}. A new one will be created.")

    # === Process each question ===
    print(f"\nProcessing {len(dataset)} questions...")
    warning_printed = False

    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, dataset_line in enumerate(dataset):
            item_id = dataset_line["id"]

            # Skip already processed items
            if item_id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items.")
                    warning_printed = True
                translated_lines.append(dataset_line)
                continue

            # Get question content
            question_content = dataset_line["question"][0][0]["content"]
            functions = dataset_line["function"]

            # Get ground truth
            if item_id not in gt_map:
                print(f"Warning: No ground truth found for {item_id}, skipping...")
                continue

            ground_truth = gt_map[item_id]

            # Create translation prompt
            prompt = create_translation_prompt(
                question_content,
                functions,
                ground_truth
            )

            # Translate using OpenAI
            print(f"\n[{i+1}/{len(dataset)}] Translating {item_id}...", end=" ")
            translated_content, _ = translate_with_openai(client, prompt)

            if translated_content:
                # Create translated item
                translated_line = dataset_line.copy()
                translated_line["question"] = [[{
                    "role": "user",
                    "content": translated_content
                }]]
                translated_lines.append(translated_line)

                print(f"✓")
                print(f"  Original: {question_content[:70]}...")
                print(f"  Hindi: {translated_content[:70]}...")

                # Save progress after each translation
                f_out.seek(0)
                f_out.truncate()

                # Sort by id before saving
                sorted_lines = sorted(
                    translated_lines,
                    key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
                )
                for t_line in sorted_lines:
                    f_out.write(json.dumps(t_line, ensure_ascii=False) + "\n")
                f_out.flush()
            else:
                print(f"✗")
                print(f"  Warning: Translation failed for {item_id}")
                exit(1)

        # Final sort and save
        sorted_lines = sorted(
            translated_lines,
            key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
        )
        f_out.seek(0)
        f_out.truncate()
        for t_line in sorted_lines:
            f_out.write(json.dumps(t_line, ensure_ascii=False) + "\n")
        f_out.flush()

    print(f"\n✅ Translation complete! Output saved to: {output_file}")
    print(f"Total items translated: {len(translated_lines)}")


if __name__ == "__main__":
    main()
