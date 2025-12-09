# Chinese Translation Code Reference Report

## Summary
The codebase has comprehensive Chinese language support with translation capabilities at multiple points in the pipeline. There are **no specific Chinese dialects** (like Traditional vs Simplified, Cantonese, etc.) explicitly handled - the code treats Chinese as a single language.

---

## 1. Configuration Files

### Location: `config.py` (Lines 54-56)
```python
class Language(Enum):
    CHINESE = auto()
    HINDI = auto()
```
- Single `CHINESE` enum (no dialect differentiation)
- No Traditional/Simplified distinction
- No regional variants

---

## 2. Tool Configuration Files

### File: `tool_config1.py` (Lines 41-50)
Configuration template for Chinese translation:
```python
Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED),
Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE),
Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_ALLOW_SYNONYM_DIFFERENT_LANGUAGE),
Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_ALLOW_SYNONYM_SAME_LANGUAGE),
Translated(language=Language.CHINESE, option=TranslateOption.PARTIALLY_TRANSLATED),
Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE_ALLOW_SYNONYM_SAME_LANGUAGE),
Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PRE_TRANSLATE),
Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_TRANSLATE),
Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PRE_TRANSLATE_ALLOW_SYNONYM_SAME_LANGUAGE),
Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_TRANSLATE_ALLOW_SYNONYM_SAME_LANGUAGE),
```

### File: `tool_config_slurm.py` (Lines 41-50)
Identical configuration template to `tool_config1.py`

---

## 3. Main Processing Pipeline

### File: `tool_main.py` (Lines 293-296)
Language tag assignment:
```python
match config.translate_mode:
    case Translated(language, option):
        match language:
            case Language.CHINESE:
                language_tag = "_zh"
```
- Maps `Language.CHINESE` → `_zh` tag
- Used in filenames: `_zh_fulltrans_nonoise_...`

---

## 4. Translation Implementation: GPT-5

### File: `models/gpt5_interface.py` (Lines 279-330)

#### Method: `async def translate_tool_question_async()`
**Purpose**: Translate Chinese questions to English before inference

```python
async def translate_tool_question_async(
    self,
    backend: ModelBackend,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Translate a user question to English using GPT-5.
    """
    messages = [
        {
            "role": "developer",
            "content": "You are a professional translator. Translate the given text to English accurately. If the given text is already in English or is language agnostic, return it unchanged."
        },
        {
            "role": "user",
            "content": f"Translate the following question to English. Only output the translated question, nothing else:\n\n{question}"
        }
    ]
    
    # Calls GPT-5 API to translate
    response = await client.responses.create(...)
    return response.output_text.strip()
```

**Key Points**:
- Uses GPT-5 as translator (hardcoded, not language-specific)
- Prompt: "Translate to English accurately"
- Handles both Chinese and already-English questions
- No language detection (assumes context)

#### Method: `async def translate_tool_answer_async()` (Lines 332-365)
**Purpose**: Translate Chinese parameter values to English after inference

```python
async def translate_tool_answer_async(
    self,
    backend: ModelBackend,
    parameter_value: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """
    Translate a single function parameter value to English using GPT-5.
    """
    # Similar structure to translate_tool_question_async
```

**Key Points**:
- Translates individual parameter values
- Same GPT-5 backend
- Used in POST_TRANSLATE phase

---

## 5. Translation Implementation: Qwen3

### File: `models/qwen3_interface.py` (Lines 245-288)

#### Method: `async def translate_tool_question_async()`
**Purpose**: Translate Chinese questions to English before inference

```python
async def translate_tool_question_async(
    self,
    backend: ModelBackend,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Translate a user question to English using Qwen3.
    """
    # Build translation prompt
    prompt_text = (
        "You are a professional translator. Translate the given text to English accurately. "
        "If the given text is already in English or is language agnostic, return it unchanged.\n\n"
        f"Translate the following question to English. Only output the translated question, nothing else:\n\n{question}"
    )
    
    # Format with ChatML (Qwen's format)
    formatted_prompt = f"<|im_start|>system\nYou are a professional translator.<|im_end|>\n"
    formatted_prompt += f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
    formatted_prompt += "<|im_start|>assistant\n"
    
    # Disable thinking mode for translation
    formatted_prompt += "<think>\n\n</think>\n\n"
    
    # Call backend
    result = await backend.generate_async(...)
    return result.generated_text.strip()
```

**Key Points**:
- Uses Qwen3 local model for translation
- ChatML format (Qwen-specific)
- Explicitly disables thinking mode (`<think></think>`)
- Same translator-agnostic prompt as GPT-5

#### Method: `async def translate_tool_answer_async()` (Lines 290-330)
**Purpose**: Translate Chinese parameter values to English

**Key Points**:
- Same ChatML format
- Thinking mode disabled
- Translates individual parameter values

---

## 6. Main Pipeline Integration

### File: `tool_main.py` - PASS 1: Pre-Translation (Lines 447-521)

**When it runs**:
- Condition: `pre_translate_tag == "_pretrans"`
- Enabled by: `TranslateOption.FULLY_TRANSLATED_PRE_TRANSLATE` or `...PRE_TRANSLATE_ALLOW_SYNONYM_SAME_LANGUAGE`

**Process**:
```python
# For each Chinese question in dataset:
translated_question = await translation_interface.translate_tool_question_async(
    backend=translation_backend,
    question=question
)

# Store translated question back into case
modified_case["question"][0][0]['content'] = translated_question
```

**Output**: `tool/result/pre_translate/{model}/_zh_fulltrans_pretrans_...json`

### File: `tool_main.py` - PASS 4: Post-Translation (Lines 810+)

**When it runs**:
- Condition: `post_translate_tag == "_posttrans"`
- Enabled by: `TranslateOption.FULLY_TRANSLATED_POST_TRANSLATE` or `...POST_TRANSLATE_ALLOW_SYNONYM_SAME_LANGUAGE`

**Process**:
```python
# For each predicted function parameter:
translated_value = await model_interface.translate_tool_answer_async(
    backend=post_translate_backend,
    parameter_value=chinese_parameter_value
)
```

**Reason**: Translate predicted values back to English for evaluation (since ground truth is in English)

---

## 7. Heatmap Generation

### File: `tool_generate_heatmap.py` (Line 94)
Chinese filename example:
```python
# Example: _zh_fulltrans_nopretrans_nonoise_noprompt_noposttrans_noallow.json
```

Maps Chinese configurations to heatmap cells:
- X-axis: Translation modes (FT, PT, PRE, POST, PREASS, POSTASS, ASS, ASD, PAR, NT)
- Y-axis: Noise modes (NO_NOISE, PARAPHRASE, SYNONYM)

---

## 8. TranslateOption Enum Definition

### File: `config.py` (Lines 59-69)
```python
class TranslateOption(Enum):
    FULLY_TRANSLATED = auto()                                          # FT
    FULLY_TRANSLATED_PROMPT_TRANSLATE = auto()                        # PT
    PARTIALLY_TRANSLATED = auto()                                      # PAR
    FULLY_TRANSLATED_ALLOW_SYNONYM_DIFFERENT_LANGUAGE = auto()        # ASD
    FULLY_TRANSLATED_ALLOW_SYNONYM_SAME_LANGUAGE = auto()             # ASS
    FULLY_TRANSLATED_PROMPT_TRANSLATE_ALLOW_SYNONYM_SAME_LANGUAGE = auto()  # PTASS
    FULLY_TRANSLATED_PRE_TRANSLATE = auto()                           # PRE
    FULLY_TRANSLATED_POST_TRANSLATE = auto()                          # POST
    FULLY_TRANSLATED_PRE_TRANSLATE_ALLOW_SYNONYM_SAME_LANGUAGE = auto()     # PREASS
    FULLY_TRANSLATED_POST_TRANSLATE_ALLOW_SYNONYM_SAME_LANGUAGE = auto()    # POSTASS
```

**No dialect-specific options** - All translations target generic "English"

---

## 9. Dataset Files

### Files with Chinese content:
```
tool/dataset/
├── BFCL_v4_multiple_zh_fulltrans_nonoise.json      # Full translation, no noise
├── BFCL_v4_multiple_zh_fulltrans_para.json         # Full translation, paraphrased
├── BFCL_v4_multiple_zh_fulltrans_syno.json         # Full translation, synonyms
├── BFCL_v4_multiple_zh_parttrans_nonoise.json      # Partial translation, no noise
├── BFCL_v4_multiple_zh_parttrans_para.json         # Partial translation, paraphrased
├── BFCL_v4_multiple_zh_parttrans_syno.json         # Partial translation, synonyms
└── possible_answer/BFCL_v4_multiple.json           # Ground truth (English)

judge/datasets/
├── zh_cn_correct.jsonl                             # Chinese correct answers
└── zh_cn_incorrect.jsonl                           # Chinese incorrect answers
```

**Note**: `zh_cn` in judge datasets suggests Simplified Chinese, but code doesn't differentiate

---

## 10. Translation Flow Summary

```
INPUT (Chinese):
  Question: "请调用函数X来回答这个问题"
  
PASS 1 (Optional - Pre-Translate):
  GPT-5/Qwen3 translates: "Please call function X to answer this question"
  
PASS 2 (Inference Raw):
  Model generates function calls with parameters in Chinese or English
  
PASS 3 (Inference JSON):
  Parse function calls
  
PASS 4 (Optional - Post-Translate):
  For each parameter value, translate back to English:
    "某个值" → "some_value"
  
PASS 5 & 6 (Evaluation & Scoring):
  Compare English predictions against English ground truth
```

---

## 11. Key Observations

### No Dialect Handling:
- ❌ No Traditional vs Simplified Chinese distinction
- ❌ No regional dialect detection (Cantonese, Mandarin, etc.)
- ❌ All "Chinese" is treated uniformly

### Translation Details:
- ✅ Question translation (pre-inference)
- ✅ Answer/parameter translation (post-inference)
- ✅ Generic English target (not locale-specific)
- ✅ Translator-agnostic prompts

### Model Support:
- ✅ GPT-5: Uses API responses
- ✅ Qwen3: Uses ChatML format
- ❌ No dialect-specific translation instructions in prompts

### Datasets:
- ✅ Multiple Chinese datasets (full, partial, noisy versions)
- ✅ Judge datasets with zh_cn label (suggests Simplified)
- ❌ No metadata about which Chinese variant

---

## 12. Code Locations Reference

| Task | File | Lines |
|------|------|-------|
| Config definition | `config.py` | 54-56, 59-69 |
| Language tagging | `tool_main.py` | 293-296 |
| Pre-translation call | `tool_main.py` | 490 |
| Post-translation call | `tool_main.py` | ~810+ |
| GPT-5 question translation | `models/gpt5_interface.py` | 279-330 |
| GPT-5 answer translation | `models/gpt5_interface.py` | 332-365 |
| Qwen3 question translation | `models/qwen3_interface.py` | 245-288 |
| Qwen3 answer translation | `models/qwen3_interface.py` | 290-330 |
| Config template | `tool_config1.py` | 41-50 |
| Config template | `tool_config_slurm.py` | 41-50 |

---

## Conclusion

**There are NO explicit Chinese dialect/variant distinctions in the codebase.** All Chinese language support treats the input as generic Chinese and translates to generic English. The `zh_cn` label in judge datasets suggests Simplified Chinese content, but this is not enforced or distinguished in the translation code itself.
