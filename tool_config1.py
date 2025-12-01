"""
Sample configuration file for tool_main.py

This file demonstrates how to configure different combinations of:
- Models (API or Local)
- Translation modes
- Noise modes

Usage:
    python tool_main.py --config tool_config1.py --num-gpus 1
"""

from config import (
    ToolConfig,
    ApiModel,
    LocalModel,
    Language,
    TranslateOption,
    AddNoiseMode,
    Translated,
    NotTranslated,
)

# Example configurations
configs = [
    # Example 1: GPT-5 with no translation, no noise (vanilla/baseline)
    ToolConfig(
        model=ApiModel.GPT_5,
        translate_mode=NotTranslated(),
        add_noise_mode=AddNoiseMode.NO_NOISE
    ),

    # Example 2: GPT-5 Mini with Chinese full translation
    ToolConfig(
        model=ApiModel.GPT_5_MINI,
        translate_mode=Translated(
            language=Language.CHINESE,
            option=TranslateOption.FULLY_TRANSLATED
        ),
        add_noise_mode=AddNoiseMode.NO_NOISE
    ),

    # Example 3: DeepSeek with Chinese translation + prompt translate
    ToolConfig(
        model=ApiModel.DEEPSEEK_CHAT,
        translate_mode=Translated(
            language=Language.CHINESE,
            option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE
        ),
        add_noise_mode=AddNoiseMode.NO_NOISE
    ),

    # Example 4: Local Qwen3-32B with Chinese translation + synonym noise
    ToolConfig(
        model=LocalModel.QWEN3_32B,
        translate_mode=Translated(
            language=Language.CHINESE,
            option=TranslateOption.FULLY_TRANSLATED
        ),
        add_noise_mode=AddNoiseMode.SYNONYM
    ),

    # Example 5: Local Qwen3-32B with Hindi partial translation + paraphrase noise
    ToolConfig(
        model=LocalModel.QWEN3_32B,
        translate_mode=Translated(
            language=Language.HINDI,
            option=TranslateOption.PARTIALLY_TRANSLATED
        ),
        add_noise_mode=AddNoiseMode.PARAPHRASE
    ),

    # Example 6: Local Granite 4.0 with Chinese + post-processing (same language)
    ToolConfig(
        model=LocalModel.GRANITE_4_0_H_SMALL,
        translate_mode=Translated(
            language=Language.CHINESE,
            option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME
        ),
        add_noise_mode=AddNoiseMode.NO_NOISE
    ),
]

# Uncomment to generate all combinations programmatically
# configs = []
# for model in [ApiModel.GPT_5, LocalModel.QWEN3_32B]:
#     for translate_mode in [
#         NotTranslated(),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED),
#         Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE),
#         Translated(language=Language.CHINESE, option=TranslateOption.PARTIALLY_TRANSLATED),
#     ]:
#         for add_noise_mode in [
#             AddNoiseMode.NO_NOISE,
#             AddNoiseMode.SYNONYM,
#             AddNoiseMode.PARAPHRASE,
#         ]:
#             configs.append(
#                 ToolConfig(
#                     model=model,
#                     translate_mode=translate_mode,
#                     add_noise_mode=add_noise_mode,
#                 )
#             )
