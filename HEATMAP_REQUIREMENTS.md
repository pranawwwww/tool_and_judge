# Datasets Required for Heatmap Generation

## Overview
To generate a heatmap showing accuracy across all translation and noise modes, you need **score files** from the tool evaluation pipeline. These are the final output after all 6 passes complete.

## Required Dataset Structure

### Location
```
tool/result/score/{MODEL_NAME}/
```

### Complete Matrix Needed
The heatmap requires score files for **all combinations** of:
- **Translation Modes**: 11 different modes
- **Noise Modes**: 3 different modes
- **Total combinations needed**: 11 × 3 = **33 score files per model**

---

## Breakdown by Translation Mode

### 1. **NT** (Not Translated)
- For English baseline (no translation needed)
- Required configurations:
  - `_en_na_nopretrans_nonoise_noprompt_noposttrans_noallow.json`
  - `_en_na_nopretrans_para_noprompt_noposttrans_noallow.json`
  - `_en_na_nopretrans_syno_noprompt_noposttrans_noallow.json`

### 2. **PAR** (Partially Translated)
- For Chinese/Hindi with partial translation
- Required configurations (example for Chinese, repeat for Hindi):
  - `_zh_parttrans_nopretrans_nonoise_noprompt_noposttrans_noallow.json`
  - `_zh_parttrans_nopretrans_para_noprompt_noposttrans_noallow.json`
  - `_zh_parttrans_nopretrans_syno_noprompt_noposttrans_noallow.json`

### 3. **FT** (Fully Translated)
- Basic fully translated without extras
- Required configurations:
  - `_zh_fulltrans_nopretrans_nonoise_noprompt_noposttrans_noallow.json`
  - `_zh_fulltrans_nopretrans_para_noprompt_noposttrans_noallow.json`
  - `_zh_fulltrans_nopretrans_syno_noprompt_noposttrans_noallow.json`

### 4. **PT** (Fully Translated + Prompt Translate)
- Fully translated + ask LLM to process in English
- Required configurations:
  - `_zh_fulltrans_nopretrans_nonoise_prompt_noposttrans_noallow.json`
  - `_zh_fulltrans_nopretrans_para_prompt_noposttrans_noallow.json`
  - `_zh_fulltrans_nopretrans_syno_prompt_noposttrans_noallow.json`

### 5. **ASD** (Allow-Synonym Different)
- Fully translated + fuzzy match synonyms across languages
- Required configurations:
  - `_zh_fulltrans_nopretrans_nonoise_noprompt_noposttrans_allowdiff.json`
  - `_zh_fulltrans_nopretrans_para_noprompt_noposttrans_allowdiff.json`
  - `_zh_fulltrans_nopretrans_syno_noprompt_noposttrans_allowdiff.json`

### 6. **ASS** (Allow-Synonym Same)
- Fully translated + fuzzy match synonyms within language
- Required configurations:
  - `_zh_fulltrans_nopretrans_nonoise_noprompt_noposttrans_allowsame.json`
  - `_zh_fulltrans_nopretrans_para_noprompt_noposttrans_allowsame.json`
  - `_zh_fulltrans_nopretrans_syno_noprompt_noposttrans_allowsame.json`

### 7. **PTASS** (Prompt Translate + Allow-Synonym Same)
- Fully translated + prompt + synonym matching
- Required configurations:
  - `_zh_fulltrans_nopretrans_nonoise_prompt_noposttrans_allowsame.json`
  - `_zh_fulltrans_nopretrans_para_prompt_noposttrans_allowsame.json`
  - `_zh_fulltrans_nopretrans_syno_prompt_noposttrans_allowsame.json`

### 8. **PRE** (Pre-Translate)
- Fully translated + translate questions before inference
- Required configurations:
  - `_zh_fulltrans_pretrans_nonoise_noprompt_noposttrans_noallow.json`
  - `_zh_fulltrans_pretrans_para_noprompt_noposttrans_noallow.json`
  - `_zh_fulltrans_pretrans_syno_noprompt_noposttrans_noallow.json`

### 9. **POST** (Post-Translate)
- Fully translated + translate answers after inference
- Required configurations:
  - `_zh_fulltrans_nopretrans_nonoise_noprompt_posttrans_noallow.json`
  - `_zh_fulltrans_nopretrans_para_noprompt_posttrans_noallow.json`
  - `_zh_fulltrans_nopretrans_syno_noprompt_posttrans_noallow.json`

### 10. **PREASS** (Pre-Translate + Allow-Synonym Same)
- Fully translated + pre-translate + synonym matching
- Required configurations:
  - `_zh_fulltrans_pretrans_nonoise_noprompt_noposttrans_allowsame.json`
  - `_zh_fulltrans_pretrans_para_noprompt_noposttrans_allowsame.json`
  - `_zh_fulltrans_pretrans_syno_noprompt_noposttrans_allowsame.json`

### 11. **POSTASS** (Post-Translate + Allow-Synonym Same)
- Fully translated + post-translate + synonym matching
- Required configurations:
  - `_zh_fulltrans_nopretrans_nonoise_noprompt_posttrans_allowsame.json`
  - `_zh_fulltrans_nopretrans_para_noprompt_posttrans_allowsame.json`
  - `_zh_fulltrans_nopretrans_syno_noprompt_posttrans_allowsame.json`

---

## Noise Modes (Applied to All Translation Modes)

Each translation mode needs 3 noise variants:

1. **NO_NOISE** (`nonoise`)
   - Original dataset without modification
   - Example: `..._nonoise_...`

2. **PARAPHRASE** (`para`)
   - Functions rephrased while keeping meaning
   - Example: `..._para_...`

3. **SYNONYM** (`syno`)
   - Parameter names replaced with synonyms
   - Example: `..._syno_...`

---

## File Format Requirements

Each score file must contain (at minimum):
```json
{
  "accuracy": 0.75,
  ...other metrics...
}
```

The heatmap reads only the `accuracy` field from the first line of each JSON file.

---

## Summary Table

| Translation Mode | Noise Modes Needed | Total Files |
|---|---|---|
| NT | 3 (nonoise, para, syno) | 3 |
| PAR | 3 (nonoise, para, syno) | 3 |
| FT | 3 (nonoise, para, syno) | 3 |
| PT | 3 (nonoise, para, syno) | 3 |
| ASD | 3 (nonoise, para, syno) | 3 |
| ASS | 3 (nonoise, para, syno) | 3 |
| PTASS | 3 (nonoise, para, syno) | 3 |
| PRE | 3 (nonoise, para, syno) | 3 |
| POST | 3 (nonoise, para, syno) | 3 |
| PREASS | 3 (nonoise, para, syno) | 3 |
| POSTASS | 3 (nonoise, para, syno) | 3 |
| **TOTAL** | | **33 files per model** |

---

## How to Run the Heatmap

Once you have all score files in `tool/result/score/{MODEL_NAME}/`:

```bash
python tool_generate_heatmap.py
```

The script will:
1. Scan all JSON files in `tool/result/score/{MODEL_NAME}/`
2. Extract accuracy from each file
3. Map filenames to translation + noise mode combinations
4. Create a visualization showing all 33 combinations
5. Save to `heatmap_{MODEL_NAME}.png`

---

## Partial Heatmap Generation

If you don't have all 33 files:
- The heatmap will show `NaN` (no data) for missing files
- It will still generate and visualize whatever data is available
- Missing files appear as blank cells in the heatmap

---

## Checklist for Complete Dataset

For **ONE model** to have a complete heatmap:

- [ ] 3 files for NT mode (nonoise, para, syno)
- [ ] 3 files for PAR mode (nonoise, para, syno)
- [ ] 3 files for FT mode (nonoise, para, syno)
- [ ] 3 files for PT mode (nonoise, para, syno)
- [ ] 3 files for ASD mode (nonoise, para, syno)
- [ ] 3 files for ASS mode (nonoise, para, syno)
- [ ] 3 files for PTASS mode (nonoise, para, syno)
- [ ] 3 files for PRE mode (nonoise, para, syno)
- [ ] 3 files for POST mode (nonoise, para, syno)
- [ ] 3 files for PREASS mode (nonoise, para, syno)
- [ ] 3 files for POSTASS mode (nonoise, para, syno)

**Total: 33 score files per model**

For multiple models, multiply by the number of models.
