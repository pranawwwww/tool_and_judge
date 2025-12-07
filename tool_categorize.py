"""
Categorization module for analyzing and grouping evaluation errors by type.

This module provides functions to categorize evaluation errors into different types
(e.g., syntax errors, wrong values, language mismatches, etc.).
"""

import json
import re
from typing import Any, Dict, Tuple
from config import ToolErrorCategory
from allow_synonym import _get_allow_synonym_backend


async def categorize_single_sample_async(evaluation_entry: Dict[str, Any]) -> Tuple[ToolErrorCategory, str]:
    """
    Asynchronously categorize a single evaluation sample to determine error type using LLM.

    Args:
        evaluation_entry: Evaluation result entry with 'id', 'valid', and error details

    Returns:
        Tuple of (ToolErrorCategory enum, raw_llm_response string)
    """
    # If the sample is valid, it's not an error
    is_valid = evaluation_entry.get("valid", False)
    # if is_valid:
    #     return ToolErrorCategory.EXACTLY_SAME_MEANING, ""
    assert not is_valid, "Expected invalid sample for categorization"

    # Prepare the prompt for LLM categorization
    system_prompt = """You are an error categorization system. Given an evaluation error entry, determine which category the error belongs to.

Here are the available categories and their definitions:
1. syntax_error: The output contains syntax errors or is not well-formed.
2. misc_errors: This error is specific to the following scenarios: function name mismatch, wrong number of functions and missing required arguments.
3. wrong_values: The output contains COMPLETELY incorrect values, calculations, or factual inaccuracies. Format and expression convention errors are NOT included in this category. If the answer is not related to math, quantity or time, and some of the words or meanings in the output overlap with the ground truth, select relevant_but_incorrect instead.
4. language_mismatch: The answer contains text that is not in English.
5. relevant_but_incorrect: The arguments are in English and relevant to the ground truth but not exactly the same in meaning.
6. exactly_same_meaning: The output is in English, and conveys the exact same meaning as the ground truth, though does not match the ground truth verbatim.
7. other_errors: The error does not fit into any of the above categories. Please try your best to avoid using this category.

For an "invalid value" error, you need to be careful about which sub-category it belongs to, since it can be one of wrong_values, language_mismatch, relevant_but_incorrect, or exactly_same_meaning.

CRITICAL: You must put your final decision inside \\boxed{} like this: \\boxed{category_name}
where category_name is exactly one of: syntax_error, misc_errors, wrong_values, language_mismatch, relevant_but_incorrect, exactly_same_meaning, or other_errors (lowercase, no extra punctuation).
If the decision is non-trivial, you can do simple reasoning before the final answer if you wish."""

    # Format the evaluation entry for the user prompt
    error_details = json.dumps(evaluation_entry, ensure_ascii=False, indent=2)

    user_prompt = f"""Evaluation error entry:
{error_details}

Based on the error details above, which category does this error belong to?
Put your final answer in \\boxed{{category_name}}."""

    # Get backend
    backend = _get_allow_synonym_backend()

    # Use the backend's client directly for chat completion
    from models.api_backend import APIBackend
    if isinstance(backend, APIBackend):
        client = backend.client
    else:
        raise TypeError(f"Expected APIBackend, got {type(backend)}")

    # Make API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = await client.chat.completions.create(
            model="gpt-5",  # Use the same model as allow_synonym
            messages=messages,
        )

        # Extract and validate response content
        if not response.choices or len(response.choices) == 0:
            print(f"Warning: LLM returned no choices for sample {evaluation_entry.get('id')}. Defaulting to other_errors.")
            return ToolErrorCategory.OTHER_ERRORS, "Error: No choices returned"

        content = response.choices[0].message.content
        if content is None:
            print(f"Warning: LLM returned None content for sample {evaluation_entry.get('id')}. Defaulting to other_errors.")
            return ToolErrorCategory.OTHER_ERRORS, "Error: None content"

        if not content.strip():
            print(f"Warning: LLM returned empty response for sample {evaluation_entry.get('id')}. Defaulting to other_errors.")
            return ToolErrorCategory.OTHER_ERRORS, "Error: Empty response"

        # Store the raw response for output
        raw_response = content

        # Extract category from \boxed{category_name} using regex
        # Pattern matches \boxed{content} and captures the content
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, content)

        if not match:
            print(f"Warning: LLM response did not contain \\boxed{{}} for sample {evaluation_entry.get('id')}. Response: {content[:100]}... Defaulting to other_errors.")
            return ToolErrorCategory.OTHER_ERRORS, raw_response

        # Extract and normalize the category name
        raw_category = match.group(1).strip().lower()

        # Map the output to ToolErrorCategory enum
        category_map = {
            "syntax_error": ToolErrorCategory.SYNTAX_ERROR,
            "misc_errors": ToolErrorCategory.MISC_ERRORS,
            "wrong_values": ToolErrorCategory.WRONG_VALUES,
            "language_mismatch": ToolErrorCategory.LANGUAGE_MISMATCH,
            "relevant_but_incorrect": ToolErrorCategory.RELEVANT_BUT_INCORRECT,
            "exactly_same_meaning": ToolErrorCategory.EXACTLY_SAME_MEANING,
            "other_errors": ToolErrorCategory.OTHER_ERRORS,
        }

        if raw_category in category_map:
            return category_map[raw_category], raw_response
        else:
            print(f"Warning: LLM returned invalid category '{raw_category}' in \\boxed{{}} for sample {evaluation_entry.get('id')}. Defaulting to other_errors.")
            return ToolErrorCategory.OTHER_ERRORS, raw_response

    except Exception as e:
        print(f"Error: Failed to categorize sample {evaluation_entry.get('id')}: {e}. Defaulting to other_errors.")
        return ToolErrorCategory.OTHER_ERRORS, f"Error: {str(e)}"
