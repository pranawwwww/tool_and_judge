# Hindi Dataset Generation Scripts

This document describes the three new Hindi dataset generation scripts created to support Hindi language evaluation in the tool project.

## Scripts Created

### 1. `generate_translated_hindi.py`
**Purpose**: Generate fully translated and partially translated Hindi datasets

**Features**:
- Generates both `FULLY_TRANSLATED` and `PARTIALLY_TRANSLATED` variants
- Creates:
  - `dataset/BFCL_v4_multiple_hi_full.json` - Fully translated to Hindi
  - `dataset/BFCL_v4_multiple_hi_partial.json` - Partially translated (preserves English keywords)
- Uses OpenAI GPT-4o-mini model
- Supports resumption - skips already translated items
- For partial translation, extracts important terms from ground truth to preserve them

**Usage**:
```bash
cd tool
python generate_translated_hindi.py
```

**Output Files**:
- `dataset/BFCL_v4_multiple_hi_full.json` (fully translated to Hindi)
- `dataset/BFCL_v4_multiple_hi_partial.json` (partially translated, English keywords preserved)

---

### 2. `generate_paraphrased_dataset_hindi.py`
**Purpose**: Generate paraphrased (rephrased) versions of Hindi datasets

**Features**:
- Takes existing Hindi datasets and creates paraphrased variants
- Processes both full and partial translation variants:
  - `BFCL_v4_multiple_hi_full.json` → `BFCL_v4_multiple_hi_full_para.json`
  - `BFCL_v4_multiple_hi_partial.json` → `BFCL_v4_multiple_hi_partial_para.json`
- Preserves meaning while rephrasing naturally in Hindi
- Keeps all English words and numbers unchanged
- Uses OpenAI GPT-4o-mini model
- Supports resumption - skips already processed items

**Usage**:
```bash
cd tool
python generate_paraphrased_dataset_hindi.py
```

**Output Files**:
- `dataset/BFCL_v4_multiple_hi_full_para.json` (paraphrased version of full translation)
- `dataset/BFCL_v4_multiple_hi_partial_para.json` (paraphrased version of partial translation)

---

### 3. `generate_synonym_dataset_hindi.py`
**Purpose**: Generate synonym-based variants of Hindi datasets

**Features**:
- Takes existing Hindi datasets and creates synonym-replaced variants
- Processes both full and partial translation variants:
  - `BFCL_v4_multiple_hi_full.json` → `BFCL_v4_multiple_hi_full_syno.json`
  - `BFCL_v4_multiple_hi_partial.json` → `BFCL_v4_multiple_hi_partial_syno.json`
- Replaces Hindi words with appropriate synonyms while preserving meaning
- Keeps all English words, numbers, and technical terms unchanged
- Uses OpenAI GPT-4o-mini model
- Supports resumption - skips already processed items

**Usage**:
```bash
cd tool
python generate_synonym_dataset_hindi.py
```

**Output Files**:
- `dataset/BFCL_v4_multiple_hi_full_syno.json` (synonym-replaced version of full translation)
- `dataset/BFCL_v4_multiple_hi_partial_syno.json` (synonym-replaced version of partial translation)

---

## Dataset Generation Pipeline

To generate all Hindi datasets, follow this order:

### Step 1: Generate Base Translations
```bash
cd tool
python generate_translated_hindi.py
```
This creates:
- `BFCL_v4_multiple_hi_full.json`
- `BFCL_v4_multiple_hi_partial.json`

### Step 2: Generate Paraphrased Variants
```bash
python generate_paraphrased_dataset_hindi.py
```
This creates:
- `BFCL_v4_multiple_hi_full_para.json`
- `BFCL_v4_multiple_hi_partial_para.json`

### Step 3: Generate Synonym Variants
```bash
python generate_synonym_dataset_hindi.py
```
This creates:
- `BFCL_v4_multiple_hi_full_syno.json`
- `BFCL_v4_multiple_hi_partial_syno.json`

---

## Final Dataset Structure

After completing all steps, you will have these Hindi datasets:

```
tool/dataset/
├── BFCL_v4_multiple_hi_full.json              # Fully translated (no noise)
├── BFCL_v4_multiple_hi_full_para.json         # Fully translated + paraphrase noise
├── BFCL_v4_multiple_hi_full_syno.json         # Fully translated + synonym noise
├── BFCL_v4_multiple_hi_partial.json           # Partially translated (no noise)
├── BFCL_v4_multiple_hi_partial_para.json      # Partially translated + paraphrase noise
└── BFCL_v4_multiple_hi_partial_syno.json      # Partially translated + synonym noise
```

This matches the expected naming convention in `tool_main.py` and provides all 6 combinations needed for Hindi evaluation.

---

## Key Differences from Chinese Scripts

| Aspect | Chinese | Hindi |
|--------|---------|-------|
| Translation Model | DeepSeek (free tier) | OpenAI GPT-4o-mini |
| Partial Translation | Uses DeepSeek reasoning with keyword preservation | Uses OpenAI with ground truth terms extraction |
| Paraphrasing | Commented-out multi-language handling | Simple single-language handling for Hindi |
| Synonyms | Handles multi-language synonym rules | Focused on Hindi synonyms only |

---

## Configuration

All scripts use OpenAI API. Ensure you have:
1. `OPENAI_API_KEY` set in your `.env` file
2. Sufficient API quota for processing all BFCL questions

---

## Notes

- All scripts support **resumption**: if interrupted, re-running will skip already-processed items
- Processing times depend on OpenAI API rate limits and dataset size
- Each script shows progress with `[current/total]` counter
- Sorted by ID before final save to ensure consistency
- All JSON output uses `ensure_ascii=False` to properly handle Hindi characters
