"""
Hindi evaluation configuration for BFCL.
Tests all 6 Hindi dataset combinations with all GPT API models.

Models included:
- GPT-5
- GPT-5-Mini
- GPT-5-Nano
- GPT-4o-Mini

Usage:
    python tool_main.py --config tool/config_hindi.py
"""

from config import (
    Config, ApiModel, LocalModel, Language, TranslateOption, AddNoiseMode,
    Translated, NotTranslated
)

# Qwen local model variations to run locally
# Choose whichever Qwen variants are available on your system or in the LocalModel enum
qwen_models = [
    LocalModel.QWEN3_8B,
    # LocalModel.QWEN3_14B,  # enable if you have this model locally
]

# All dataset combinations (6 total)
dataset_combinations = [
    (TranslateOption.FULLY_TRANSLATED, AddNoiseMode.NO_NOISE),
    (TranslateOption.FULLY_TRANSLATED, AddNoiseMode.PARAPHRASE),
    (TranslateOption.FULLY_TRANSLATED, AddNoiseMode.SYNONYM),
    (TranslateOption.PARTIALLY_TRANSLATED, AddNoiseMode.NO_NOISE),
    (TranslateOption.PARTIALLY_TRANSLATED, AddNoiseMode.PARAPHRASE),
    (TranslateOption.PARTIALLY_TRANSLATED, AddNoiseMode.SYNONYM),
]

# Generate configs for each model and dataset combination
configs = []
for model in qwen_models:
    for translate_option, noise_mode in dataset_combinations:
        configs.append(
            Config(
                model=model,
                translate_mode=Translated(language=Language.HINDI, option=translate_option),
                add_noise_mode=noise_mode
            )
        )

