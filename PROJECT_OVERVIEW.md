# Tool and Judge Project Overview

## Project Purpose

This project consists of two main components:
1. **Tool Project**: Evaluates multilingual function calling capabilities of LLMs
2. **Judge Project**: Studies the relationship between perplexity and LLM preference judgments

---

## Architecture Overview

```
tool_and_judge/
├── tool/                          # Function calling evaluation framework
│   ├── main.py                   # Main entry point (6 passes of processing)
│   ├── dataset/                  # Input datasets for evaluation
│   ├── result/                   # Output results across different passes
│   └── models/                   # Model-specific interfaces
├── judge/                        # Preference and perplexity evaluation
│   ├── run.py / run.slurm       # Entry point for judge runs
│   ├── models/                   # Model interfaces (Granite, Qwen3)
│   ├── datasets/                 # Generated answer datasets
│   └── result/                   # Results (perplexity, preferences, metrics)
├── models/                       # Shared unified model interfaces (refactored)
├── config.py                     # Shared configuration
├── tool_main.py / judge_run.py  # Entry scripts
└── util.py / allow_synonym.py   # Utilities
```

---

## Tool Project: Function Calling Evaluation

### Overview
Evaluates how well LLMs can call functions in different languages (English, Chinese, Hindi) with various processing options (translation, noise, synonym matching).

### Configuration
**File**: `config.py` → `ToolConfig` and `TranslateOption`, `AddNoiseMode`

```python
@dataclass(frozen=True)
class ToolConfig:
    model: ToolModel  # ApiModel or LocalModel
    translate_mode: TranslateMode  # Translated or NotTranslated
    add_noise_mode: AddNoiseMode  # NO_NOISE, SYNONYM, PARAPHRASE
```

### Entry Point
**File**: `tool_main.py`
- Requires `--config` argument (e.g., `python tool_main.py --config config1.py`)
- Supports multiple configurations per run

### Processing Pipeline: 6 Passes

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Tool Processing Pipeline (6 Passes)              │
└─────────────────────────────────────────────────────────────────────┘

INPUT DATASET: tool/dataset/BFCL_v4_multiple{language}{level}{noise}.json
                └─ Multilingual function calling dataset with options

PASS 1: PRE-TRANSLATION (Optional)
├─ Condition: translate_mode = FULLY_TRANSLATED_PRE_TRANSLATE
├─ Input: Raw dataset (original language questions)
├─ Process: Translate questions to English before inference
├─ Output: tool/result/pre_translate/{model}/{tags}.json
└─ Used by: Allow-synonym pass (when enabled)

PASS 2: INFERENCE RAW
├─ Input: Dataset (or pre-translated if enabled)
├─ Process: Call LLM for function calling
│   - Generate prompts with functions and questions
│   - Model predicts function calls
│   - Raw output captured (varies by model format)
├─ Output: tool/result/inference_raw/{model}/{tags}.json
│   └─ {"id": ..., "result": "<raw_model_output>"}
└─ Uses: Model interface for generation

PASS 3: INFERENCE JSON
├─ Input: inference_raw results
├─ Process: Parse & postprocess raw outputs
│   - Model-specific parsing (API models use AST parsing)
│   - Local models use JSON/tag parsing
│   - Function name sanitization (if needed)
├─ Output: tool/result/inference_json/{model}/{tags}.json
│   └─ {"id": ..., "result": [...parsed_functions...]}
└─ Uses: Model interface postprocess_tool_calls()

PASS 4: POST-TRANSLATION (Optional)
├─ Condition: translate_mode = FULLY_TRANSLATED_POST_TRANSLATE
├─ Input: inference_json results
├─ Process: Translate function parameter values back to original language
├─ Output: tool/result/post_translate/{model}/{tags}.json
└─ Note: Runs BEFORE allow_synonym for consistency

PASS 5: ALLOW SYNONYM MATCHING (Optional)
├─ Condition: translate_mode contains ALLOW_SYNONYM
├─ Input: inference_json OR post_translate (if enabled)
├─ Process: Fuzzy match predicted parameters to acceptable synonyms
│   - Cache hits/misses tracking
│   - Separate caches for different languages
├─ Output: tool/result/allow_synonym/{model}/{tags}.json
└─ Effect: Improves accuracy by accepting equivalent answers

PASS 6: EVALUATION & SCORING
├─ Input: Final processed results (before scoring)
├─ Process 1 - EVALUATION:
│   - Compare predicted vs ground truth function calls
│   - Check structure, parameters, parameter types
│   - Output: tool/result/evaluation/{model}/{tags}.json
│       └─ Detailed per-sample evaluation results
├─ Process 2 - SCORING:
│   - Aggregate evaluation metrics
│   - Calculate accuracy, precision, recall
│   - Output: tool/result/score/{model}/{tags}.json
│       └─ Summary statistics and per-category breakdown
└─ Metrics: Accuracy, True Positives, False Positives, etc.
```

### Key Tagging System
Results stored with tags: `{language}_{level}_{pre}_{noise}_{prompt}_{post}_{synonym}`

Examples:
- `_zh_fulltrans_nopretrans_nonoise_noprompt_noposttrans_noallow.json` - Chinese, full translation, no noise
- `_en_na_nopretrans_nonoise_noprompt_noposttrans_noallow.json` - English baseline

### Supported Models

**API Models**:
- GPT-5, GPT-5-Mini, GPT-5-Nano
- GPT-4o-Mini
- DeepSeek Chat
- Llama 3.1 (8B, 70B) via AWS Bedrock

**Local Models**:
- Granite: 3.1-8B, 4.0-H-Tiny, 4.0-H-Small
- Qwen: 3-8B, 3-14B, 3-30B, 3-32B, 3-Next-80B

### Model Interfaces
**Location**: `models/` and `tool/models/`

Pattern: Each model has dedicated interface class inheriting from `ToolModelInterface`
- `Granite3.1Interface` - JSON-based tool call format
- `Qwen3Interface` - `<tool_call>{...}</tool_call>` format
- `Qwen2.5Interface` - JSON-based format
- API models - AST parsing from response text

---

## Judge Project: Preference & Perplexity Evaluation

### Overview
Compares two evaluation methods:
1. **Perplexity Method**: Use model's probability distribution to judge answer quality
2. **Preference Method**: Direct comparison asking which answer is better

Studies correlation between these methods and bias in different language pairs.

### Configuration
**File**: `config.py` → `JudgeConfig`

```python
@dataclass(frozen=True)
class JudgeConfig:
    model: LocalModel  # Local model only (needs tokenizer access)
    lang1: str  # Language code (e.g., "en", "zh_cn")
    lang2: str  # Language code
    result_type: ResultType  # PERPLEXITY, PREFERENCE_DIRECT, PREFERENCE_COT
```

### Entry Point
**File**: `judge_run.py`
- Requires `--config` argument pointing to judge config file
- Supports `--num-gpus` parameter

### Dataset Generation Pipeline

```
judge/generate_dataset.py:

BFCL Dataset (Multiple Choice Questions)
    ↓
Extract pairs by language:
    - lang1_correct.jsonl (correct answers in language 1)
    - lang1_incorrect.jsonl (incorrect answers in language 1)
    - lang2_correct.jsonl (correct answers in language 2)
    - lang2_incorrect.jsonl (incorrect answers in language 2)

Each entry structure:
{
    'index': int,
    'question': str (always English),
    'answer': str (in target language),
    'lang': str,
    'is_correct': bool,
    'subject': str
}
```

### Evaluation Types

#### 1. **Perplexity Method**
- **Process**: Calculate probability of answers given questions
- **Function**: `collect_perplexity_local_async()`
- **Output fields**:
  - `perplexity`: Average negative log probability
  - `logits`: Raw logits from model
  - `answer_tokens`: Extracted answer token sequence

- **Calculation**:
  ```
  1. Build prompt: "[Question] [Answer instruction]"
  2. Get model logits for the full prompt
  3. Extract answer tokens from token sequence
  4. Calculate negative log probability of answer tokens
  5. Perplexity = exp(avg_negative_log_prob)
  ```

#### 2. **Preference Direct (Immediate)**
- **Process**: Ask model to judge which answer is better WITHOUT reasoning
- **Function**: `collect_preference_local_direct_async()`
- **Output fields**:
  - `preference`: 1 or 2 (which answer is better)
  - `raw_output`: Full model response
  - `logit_1`: Probability/logit for answer 1
  - `logit_2`: Probability/logit for answer 2
  - `error`: Any error message

- **Prompt Format**:
  ```
  Given the following question and two answers, which answer is better?
  Question: [question]
  Answer 1: [answer1]
  Answer 2: [answer2]
  
  Provide your judgment IMMEDIATELY.
  Format: \boxed{X} where X is 1 or 2
  ```

#### 3. **Preference CoT (Chain-of-Thought)**
- **Process**: Ask model to judge WITH reasoning first
- **Function**: `collect_preference_local_cot_async()`
- **Supports**: Extended thinking modes (enables reasoning)
- **Output fields**:
  - `preference`: 1 or 2
  - `raw_output`: Reasoning + judgment
  - `error`: Any error message

### Pair Combinations Evaluated
The judge evaluates 4 answer pair types:
1. **lang1_correct vs lang2_incorrect** - Should prefer lang1
2. **lang1_incorrect vs lang2_correct** - Should prefer lang2
3. **Both correct** - Either answer valid (studies language bias)
4. **Both incorrect** - Neither answer valid (error case)

### Result Comparison & Analysis
**Function**: `compare_results()` in `judge_run.py`

Metrics calculated:
```
Preference Method Results:
  - % preferring lang1 vs lang2
  - Per-category breakdown

Perplexity Method Results:
  - % lower perplexity for lang1 vs lang2
  - Per-category breakdown

Correlation Analysis:
  - Pearson correlation (measures linear relationship)
  - Spearman correlation (measures rank relationship)
  - Agreement percentage

Interpretation:
  - High correlation: Perplexity is good proxy for preference
  - Bias detection: Asymmetry between lang1→lang2 vs lang2→lang1
```

### File Organization
```
judge/
├── datasets/
│   ├── en_correct.jsonl
│   ├── en_incorrect.jsonl
│   ├── zh_cn_correct.jsonl
│   └── zh_cn_incorrect.jsonl
└── result/
    └── {model_name}/
        ├── preferences_local_direct/
        │   └── {lang1}_{type1}_{lang2}_{type2}.jsonl
        ├── preferences_local_cot/
        │   └── {lang1}_{type1}_{lang2}_{type2}.jsonl
        └── perplexities_local/
            └── {lang}_{type}.jsonl
```

---

## Unified Model Interface System (Refactored)

### Base Classes
**Location**: `models/base.py`

```python
class ModelInterface:
    """Abstract base for all model behavior"""
    - get_model_name()
    - get_system_message()
    - get_assistant_prefix()

class ToolModelInterface(ModelInterface):
    """Tool calling specific"""
    - generate_tool_call_async(backend, functions, user_query, ...)
    - postprocess_tool_calls(raw_output, ...)
    - translate_tool_question_async(backend, question)

class JudgeModelInterface(ModelInterface):
    """Judgment specific"""
    - forward_for_logits_async(backend, question, answer, ...)
    - compare_directly_async(backend, question, answer1, answer2)
    - compare_thinking_async(backend, question, answer1, answer2)
```

### Model-Specific Implementations

**Qwen3** (`models/qwen3_interface.py`):
- Inherits: `JudgeModelInterface` + `ToolModelInterface`
- Special format: `<tool_call>{json}</tool_call>`
- Supports: Extended thinking mode for reasoning
- Chat template: ChatML format with special tokens

**Granite** (`judge/models/granite.py`, `tool/models/granite_interface.py`):
- Inherits: `JudgeModelInterface` + `ToolModelInterface`
- Format: JSON output in plain text
- Batch processing: Optimized for multiple samples
- No extended thinking support

**API Models** (GPT, Claude, DeepSeek):
- AST parsing of function calls from text
- Sequential processing (no batching)
- Name sanitization for some models (GPT-5)

### Factory Functions
**Location**: `models/model_factory.py` and `models/factory.py`

```python
def create_backend(backend_type, model_name, device, num_gpus, ...):
    """Create or get cached model backend"""
    # Returns: AsyncModelBackend instance

def create_interface(model_name):
    """Create model interface for infer and postprocessing"""
    # Returns: ModelInterface subclass instance
```

### Caching Strategy
- **Backend caching**: One instance per model/instance_name pair
- **Interface caching**: Created fresh per config (lightweight)
- **Instance names**: "default", "experiment", "allow_synonym" for different use cases

---

## Data Flow Examples

### Example 1: Tool Evaluation with Chinese Functions
```
Config: ToolConfig(
    model=LocalModel.QWEN3_8B,
    translate_mode=Translated(Language.CHINESE, TranslateOption.FULLY_TRANSLATED),
    add_noise_mode=AddNoiseMode.NO_NOISE
)

Pipeline:
1. Load: BFCL_v4_multiple_zh_fulltrans_nonoise.json
2. Pre-translate: (skipped, not enabled)
3. Inference raw: Qwen3 generates <tool_call>{...}</tool_call>
4. Inference JSON: Parse tool calls to structured format
5. Post-translate: (skipped, not enabled)
6. Allow synonym: (skipped, not enabled)
7. Evaluation: Compare to ground truth
8. Score: Calculate accuracy metrics

Output: tool/result/score/Qwen-Qwen3-8B/_zh_fulltrans_nopretrans_nonoise_noprompt_noposttrans_noallow.json
```

### Example 2: Judge Evaluation with Perplexity
```
Config: JudgeConfig(
    model=LocalModel.QWEN3_8B,
    lang1="zh_cn",
    lang2="en",
    result_type=ResultType.PERPLEXITY
)

Pipeline:
1. Generate datasets: Create {zh_cn,en}_{correct,incorrect}.jsonl
2. Individual entry analysis: Calculate perplexity for each answer
   - Prompt: "Question [in English]: [Q]. Please answer in [language]..."
   - Get logits from model's hidden states
   - Calculate probability of answer tokens
   - Convert to perplexity

3. Result comparison:
   - Compare perplexity ranks with preference judgments
   - Calculate correlation between methods
   - Detect language bias

Output: judge/result/Qwen-Qwen3-8B/perplexities_local/zh_cn_correct.jsonl
```

---

## Configuration Examples

### Tool Project Config (config1.py)
```python
configs = [
    ToolConfig(
        model=LocalModel.QWEN3_8B,
        translate_mode=Translated(Language.CHINESE, TranslateOption.FULLY_TRANSLATED),
        add_noise_mode=AddNoiseMode.NO_NOISE
    ),
    ToolConfig(
        model=ApiModel.GPT_4O_MINI,
        translate_mode=NotTranslated(),
        add_noise_mode=AddNoiseMode.SYNONYM
    ),
]
```

### Judge Config (judge_config1.py)
```python
configs = [
    JudgeConfig(
        model=LocalModel.QWEN3_8B,
        lang1="en",
        lang2="zh_cn",
        result_type=ResultType.PREFERENCE_DIRECT
    ),
    JudgeConfig(
        model=LocalModel.QWEN3_8B,
        lang1="en",
        lang2="zh_cn",
        result_type=ResultType.PERPLEXITY
    ),
]
```

---

## Key Utilities

### Name Mapping (`models/name_mapping.py`)
Sanitizes function names for models that don't support special characters:
- GPT-5 family models require function names without dots/slashes
- Maps original names to sanitized versions and back
- Used during inference parsing and evaluation

### Allow Synonym Matching (`allow_synonym.py`)
Post-processing step to match predicted parameters with acceptable alternatives:
- Caches fuzzy matching results
- Separate handling for same/different language matching
- Improves accuracy for multilingual evaluation

### Backend Abstraction
- **HuggingFace Backend**: Direct model inference with transformers
- **vLLM Backend**: Optimized inference with vLLM engine
- **API Backend**: OpenAI/Bedrock/other API calls
- All present same async interface

---

## Output Format Standards

### Tool Results
```json
{
  "id": "sample_123",
  "result": [
    {
      "name": "function_name",
      "arguments": {
        "param1": "value1",
        "param2": "value2"
      }
    }
  ]
}
```

### Judge Results - Perplexity
```json
{
  "index": 0,
  "question": "...",
  "answer": "...",
  "lang": "en",
  "is_correct": true,
  "perplexity": 3.45,
  "logits": [...],
  "model": "Qwen3-8B"
}
```

### Judge Results - Preference
```json
{
  "index": 0,
  "preference": 1,
  "raw_output": "...",
  "logit_1": 0.78,
  "logit_2": 0.22,
  "error": null,
  "question": "...",
  "answer1": "...",
  "answer2": "...",
  "lang1": "en",
  "lang2": "zh_cn"
}
```

---

## Summary

**Tool Project**: Evaluates function calling across languages with advanced processing (translation, noise, synonym matching)

**Judge Project**: Studies perplexity vs preference methods for ranking answer quality

**Unified Architecture**: Shared model interfaces support both projects with minimal duplication

**Extensibility**: New models added by creating interface class + registering in factory
