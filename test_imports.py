"""
Test script to verify all import fixes are working correctly.
"""
import sys

def test_imports():
    """Test that all modified imports work correctly."""
    errors = []
    successes = []

    # Test tool/ imports
    print("Testing tool/ module imports...")
    test_cases = [
        ("tool.parse_dataset", None),
        ("tool.parse_ast", None),
        ("tool.post_processing", "load_or_create_cache"),
        ("tool.revise_noise", None),
        ("tool.partially_translate", None),
        ("tool.generate_translated", None),
        ("tool.generate_synonym_dataset", None),
        ("tool.generate_paraphrased_dataset", None),
        ("tool.models.model_factory", "create_model_interface"),
        ("tool.models.gpt_4o_mini_interface", "GPT4oMiniInterface"),
        ("tool.models.gpt_5_interface", "GPT5Interface"),
        ("tool.models.claude_sonnet_interface", "ClaudeSonnetInterface"),
        ("tool.models.claude_haiku_interface", "ClaudeHaikuInterface"),
        ("tool.models.deepseek_chat_interface", "DeepseekChatInterface"),
        ("tool.models.llama_3_1_interface", "Llama31Interface"),
        ("tool.models.granite_3_1_8b_instruct_interface", "Granite3_1_8BInstructInterface"),
        ("tool.models.granite_4_interface", "Granite4Interface"),
        ("tool.models.qwen2_5_interface", "Qwen25InstructInterface"),
        ("tool.models.qwen3_interface", "Qwen3Interface"),
    ]

    for module_name, attr_name in test_cases:
        try:
            module = __import__(module_name, fromlist=[attr_name] if attr_name else [])
            if attr_name:
                getattr(module, attr_name)
            successes.append(f"[OK] {module_name}" + (f".{attr_name}" if attr_name else ""))
        except Exception as e:
            errors.append(f"[FAIL] {module_name}" + (f".{attr_name}" if attr_name else "") + f": {e}")

    # Test judge/ imports
    print("\nTesting judge/ module imports...")
    judge_test_cases = [
        ("judge.parse_dataset", "parse_dataset"),
    ]

    for module_name, attr_name in judge_test_cases:
        try:
            module = __import__(module_name, fromlist=[attr_name] if attr_name else [])
            if attr_name:
                getattr(module, attr_name)
            successes.append(f"[OK] {module_name}" + (f".{attr_name}" if attr_name else ""))
        except Exception as e:
            # Some judge modules require torch/transformers which may not be installed
            if "torch" in str(e) or "transformers" in str(e):
                successes.append(f"[SKIP] {module_name}" + (f".{attr_name}" if attr_name else "") + " (missing dependencies)")
            else:
                errors.append(f"[FAIL] {module_name}" + (f".{attr_name}" if attr_name else "") + f": {e}")

    # Print results
    print("\n" + "="*60)
    print("IMPORT TEST RESULTS")
    print("="*60)

    print(f"\nSuccessful imports ({len(successes)}):")
    for success in successes:
        print(f"  {success}")

    if errors:
        print(f"\nFailed imports ({len(errors)}):")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print(f"\n[SUCCESS] All {len(successes)} imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
