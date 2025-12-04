import json
import os
import asyncio


async def collect_preference_local_cot_async(
        pairs,
        backend,
        model_interface,
        batch_size=8):
    """
    Use a local LLM to judge which answer is better with reasoning.

    This function uses concurrent async requests to prompt the model to analyze
    and explain its reasoning before making a decision.

    Args:
        pairs: List of question-answer pairs
        backend: AsyncModelBackend instance (HuggingFace or vLLM)
        model_interface: ModelInterface instance for model-specific behavior
        batch_size: Number of concurrent requests (default: 8)

    Returns:
        List of results to be written to file
    """

    model_name = getattr(backend, 'model_name', 'unknown')

    print(f"\nCollecting preferences with reasoning using local LLM")
    print(f"Concurrent requests: {batch_size}")
    print(f"Samples to process: {len(pairs)}")

    # Process samples with concurrency control
    semaphore = asyncio.Semaphore(batch_size)
    results = []
    processed_count = 0

    async def process_single_sample(i, pair):
        """Process a single sample asynchronously."""
        nonlocal processed_count

        async with semaphore:
            try:
                # Use the new interface to compare answers with reasoning
                comparison_result = await model_interface.compare_thinking_async(
                    backend=backend,
                    question=pair['question'],
                    answer1=pair['answer1'],
                    answer2=pair['answer2']
                )

                preference = comparison_result.preference
                raw_output = comparison_result.raw_output  # Reasoning is merged with raw output
                error = comparison_result.error

                output_result = {
                    'index': i,
                    'preference': preference,
                    'raw_output': raw_output,
                    'error': error,
                    'question': pair['question'],
                    'answer1': pair['answer1'],
                    'answer2': pair['answer2'],
                    'lang1': pair['lang1'],
                    'lang2': pair['lang2'],
                    'subject': pair.get('subject', ''),
                    'model': model_name
                }

                return output_result

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                raise

    # Create all tasks
    tasks = [process_single_sample(i, pair) for i, pair in enumerate(pairs)]

    # Process results as they complete
    for coro in asyncio.as_completed(tasks):
        result = await coro
        processed_count += 1
        results.append(result)

        if processed_count % 5 == 0 or processed_count == len(pairs):
            print(f"  Processed {processed_count}/{len(pairs)} samples")

    return results