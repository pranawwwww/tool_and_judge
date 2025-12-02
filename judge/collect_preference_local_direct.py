import json
import os
import asyncio
import torch

from config import ResultType


def collect_preference_local_direct(
        pairs,
        backend,
        model_interface,
        output_file="preferences_local.jsonl",
        batch_size=8):
    """
    Use a local LLM to judge which answer is better by comparing logits.

    This function uses concurrent async requests to the model backend to generate
    responses. The preference is determined by comparing the log probabilities
    of tokens "1" and "2" at the position immediately after the "\boxed{" prefix.

    The approach:
    1. Generate response with the prompt asking for \boxed{X} format
    2. Tokenize "\boxed{" to find the prefix length
    3. Extract logits at position prefix_length (the answer position)
    4. Compare log probabilities of tokens "1" and "2"
    5. Select the token with higher log probability as the preference

    The generated text is saved for reference but not used for decision-making.
    Errors are tracked if logit extraction fails.

    Args:
        pairs: List of question-answer pairs
        backend: AsyncModelBackend instance (HuggingFace or vLLM)
        model_interface: ModelInterface instance for model-specific behavior
        output_file: Output file for results
        batch_size: Number of concurrent requests (default: 8)

    Returns:
        None (results are written to output_file)
    """

    # Run async implementation
    asyncio.run(_collect_preference_local_direct_async(
        pairs=pairs,
        backend=backend,
        model_interface=model_interface,
        output_file=output_file,
        batch_size=batch_size
    ))


async def _collect_preference_local_direct_async(
        pairs,
        backend,
        model_interface,
        output_file,
        batch_size):
    """Async implementation of collect_preference_local_direct."""

    tokenizer = backend.tokenizer
    model_name = getattr(backend, 'model_name', 'unknown')

    # Load already processed samples if file exists
    processed_indices = set()
    results_dict = {}

    if os.path.exists(output_file):
        print(f"Loading existing results from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    idx = result['index']
                    processed_indices.add(idx)
                    results_dict[idx] = result['preference']
        print(f"Found {len(processed_indices)} already processed samples")
    if len(processed_indices) == len(pairs):
        print("All samples already processed. Exiting.")
        return

    print(f"\nCollecting preferences using local LLM")
    print(f"Results will be written to {output_file}")
    print(f"Concurrent requests: {batch_size}")

    # Collect unprocessed samples
    unprocessed_samples = []
    for i, pair in enumerate(pairs):
        if i not in processed_indices:
            unprocessed_samples.append((i, pair))

    total_to_process = len(unprocessed_samples)
    print(f"Samples to process: {total_to_process}")

    # Process samples with concurrency control
    semaphore = asyncio.Semaphore(batch_size)
    lock = asyncio.Lock()
    processed_count = 0

    async def process_single_sample(i, pair):
        """Process a single sample asynchronously."""
        nonlocal processed_count

        async with semaphore:
            try:
                # Use the new interface to compare answers directly
                comparison_result = await model_interface.compare_directly_async(
                    backend=backend,
                    question=pair['question'],
                    answer1=pair['answer1'],
                    answer2=pair['answer2']
                )

                preference = comparison_result.preference
                error = None
                log_prob_1 = None
                log_prob_2 = None
                log_prob_diff = None
                generated_text = ""

                # Write result
                output_result = {
                    'index': i,
                    'preference': preference,
                    'generated_text': generated_text,
                    'error': error,
                    'log_prob_1': log_prob_1,
                    'log_prob_2': log_prob_2,
                    'log_prob_diff': log_prob_diff,
                    'question': pair['question'],
                    'answer1': pair['answer1'],
                    'answer2': pair['answer2'],
                    'lang1': pair['lang1'],
                    'lang2': pair['lang2'],
                    'subject': pair.get('subject', ''),
                    'model': model_name
                }

                async with lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(output_result, ensure_ascii=False) + '\n')
                        f.flush()

                    results_dict[i] = preference
                    processed_count += 1

                    if processed_count % 10 == 0 or processed_count == total_to_process:
                        print(f"  Processed {processed_count}/{total_to_process} samples")

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                raise

    # Process all unprocessed samples concurrently
    tasks = [process_single_sample(i, pair) for i, pair in unprocessed_samples]
    await asyncio.gather(*tasks)

    print("\nPreference collection completed.")
