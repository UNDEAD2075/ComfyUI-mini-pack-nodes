import json
import logging
import math

import os
import random
import importlib.util
from pathlib import Path

import torch

import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management

from comfy.taesd.taesd import TAESD

LOGGER = logging.getLogger("ComfyUI-mini-pack-nodes")

NONE_LORA = "[None]"
MAX_DIMENSION = 8192

# ============================================================
# Общие функции
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()

def join_prompt_parts(parts, separator=", "):
    result = []

    for part in parts:
        part = clean_text(part)

        if part:
            result.append(part)

    return separator.join(result)

def round_dimension(value, multiple=8, mode="nearest"):
    value = max(int(value), int(multiple))

    if mode == "floor":
        result = (value // multiple) * multiple

    elif mode == "ceil":
        result = ((value + multiple - 1) // multiple) * multiple

    else:
        result = int(math.floor(value / multiple + 0.5)) * multiple

    return max(result, multiple)

def deduplicate_words(words):
    result = []
    known = set()

    for word in words:
        word = clean_text(word)

        if not word:
            continue

        key = word.casefold()

        if key not in known:
            known.add(key)
            result.append(word)

    return result

def parse_json_value(value):
    if not isinstance(value, str):
        return value

    text = value.strip()

    if not text:
        return ""

    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            pass

    return value

def value_to_words(value):
    value = parse_json_value(value)

    if value is None:
        return []

    if isinstance(value, dict):
        result = []

        for key, nested_value in value.items():
            if isinstance(nested_value, (int, float)):
                result.append((str(key), float(nested_value)))
            else:
                result.extend(value_to_words(nested_value))

        if result and all(isinstance(item, tuple) for item in result):
            result.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            return [item[0] for item in result]

        return [str(item) for item in result]

    if isinstance(value, (list, tuple, set)):
        result = []

        for item in value:
            result.extend(value_to_words(item))

        return result

    text = str(value).strip()

    if not text:
        return []

    text = text.replace(";", ",")
    text = text.replace("\n", ",")

    if "," in text:
        return [
            item.strip()
            for item in text.split(",")
            if item.strip()
        ]

    return [text]

def extract_trigger_words(metadata, max_words=32):
    if not metadata:
        return []

    result = []

    preferred_keys = (
        "ss_trigger_words",
        "trigger_words",
        "trigger_word",
        "modelspec.trigger_phrase",
        "modelspec.trigger_words",
        "tags",
    )

    for key in preferred_keys:
        if key in metadata:
            result.extend(value_to_words(metadata[key]))

    # Запасной вариант для Kohya metadata.
    if not result and "ss_tag_frequency" in metadata:
        tag_frequency = parse_json_value(
            metadata["ss_tag_frequency"]
        )

        weighted = {}

        def collect_frequency(value):
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    if isinstance(nested_value, (int, float)):
                        weighted[str(key)] = (
                            weighted.get(str(key), 0.0)
                            + float(nested_value)
                        )
                    else:
                        collect_frequency(nested_value)

        collect_frequency(tag_frequency)

        result = [
            key
            for key, _weight in sorted(
                weighted.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

    result = deduplicate_words(result)

    if max_words > 0:
        result = result[:max_words]

    return result

def read_lora_metadata(path):
    """
    Пытается прочитать только metadata safetensors,
    не загружая веса LoRA в память модели.
    """

    if path.lower().endswith(
        (".safetensors", ".sft")
    ):
        try:
            from safetensors import safe_open

            with safe_open(
                path,
                framework="pt",
                device="cpu",
            ) as handle:
                return handle.metadata() or {}

        except Exception as error:
            LOGGER.debug(
                "Metadata read failed for %s: %s",
                path,
                error,
            )

    try:
        loaded = comfy.utils.load_torch_file(
            path,
            safe_load=True,
            return_metadata=True,
        )

        if isinstance(loaded, tuple):
            _state_dict, metadata = loaded
            return metadata or {}

    except Exception as error:
        LOGGER.debug(
            "Fallback metadata read failed for %s: %s",
            path,
            error,
        )

    return {}

def load_lora_state_dict(path):
    try:
        loaded = comfy.utils.load_torch_file(
            path,
            safe_load=True,
            return_metadata=True,
        )
    except TypeError:
        loaded = comfy.utils.load_torch_file(
            path,
            safe_load=True,
        )

    if isinstance(loaded, tuple):
        return loaded[0], loaded[1] or {}

    return loaded, {}

def apply_lora(
    model,
    clip,
    lora_state_dict,
    metadata,
    strength_model,
    strength_clip,
):
    """
    Использует официальный механизм ComfyUI.
    Совместим с версиями, где lora_metadata
    ещё отсутствует в сигнатуре.
    """

    try:
        return comfy.sd.load_lora_for_models(
            model,
            clip,
            lora_state_dict,
            strength_model,
            strength_clip,
            lora_metadata=metadata,
        )

    except TypeError:
        return comfy.sd.load_lora_for_models(
            model,
            clip,
            lora_state_dict,
            strength_model,
            strength_clip,
        )

# ============================================================
# 1. Resolution Preset
# ============================================================

RESOLUTION_PRESETS = {
    "SDXL | portrait | 640x1024": (
        640,
        1024,
        "SDXL",
    ),
    "SDXL | portrait | 768x1152": (
        768,
        1152,
        "SDXL",
    ),
    "SDXL | portrait | 832x1216": (
        832,
        1216,
        "SDXL",
    ),
    "SDXL | landscape | 1024x640": (
        1024,
        640,
        "SDXL",
    ),
    "SDXL | landscape | 1152x768": (
        1152,
        768,
        "SDXL",
    ),
    "SDXL | landscape | 1216x832": (
        1216,
        832,
        "SDXL",
    ),
    "SDXL | square | 1024x1024": (
        1024,
        1024,
        "SDXL",
    ),

    "SD 1.5 | square | 512x512": (
        512,
        512,
        "SD 1.5",
    ),
    "SD 1.5 | portrait | 512x768": (
        512,
        768,
        "SD 1.5",
    ),
    "SD 1.5 | portrait | 576x864": (
        576,
        864,
        "SD 1.5",
    ),
    "SD 1.5 | portrait | 640x960": (
        640,
        960,
        "SD 1.5",
    ),
    "SD 1.5 | portrait | 640x1024": (
        640,
        1024,
        "SD 1.5",
    ),
    "SD 1.5 | landscape | 768x512": (
        768,
        512,
        "SD 1.5",
    ),
    "SD 1.5 | landscape | 960x640": (
        960,
        640,
        "SD 1.5",
    ),
    "SD 1.5 | landscape | 1024x640": (
        1024,
        640,
        "SD 1.5",
    ),

    "custom": (
        640,
        1024,
        "custom",
    ),
}

class LocalResolutionPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    list(RESOLUTION_PRESETS.keys()),
                    {
                        "default": (
                            "SDXL | portrait | 640x1024"
                        ),
                    },
                ),
                "custom_width": (
                    "INT",
                    {
                        "default": 640,
                        "min": 64,
                        "max": MAX_DIMENSION,
                        "step": 8,
                    },
                ),
                "custom_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": MAX_DIMENSION,
                        "step": 8,
                    },
                ),
                "round_to": (
                    "INT",
                    {
                        "default": 8,
                        "min": 8,
                        "max": 128,
                        "step": 8,
                    },
                ),
                "round_mode": (
                    [
                        "nearest",
                        "floor",
                        "ceil",
                    ],
                    {
                        "default": "nearest",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "STRING",
    )

    RETURN_NAMES = (
        "width",
        "height",
        "latent_width",
        "latent_height",
        "aspect_ratio",
        "info",
    )

    FUNCTION = "make_resolution"
    CATEGORY = "Local Tools/Resolution"

    def make_resolution(
        self,
        preset,
        custom_width,
        custom_height,
        round_to,
        round_mode,
    ):
        if preset == "custom":
            width = custom_width
            height = custom_height
            family = "custom"

        else:
            width, height, family = (
                RESOLUTION_PRESETS[preset]
            )

        width = round_dimension(
            width,
            round_to,
            round_mode,
        )

        height = round_dimension(
            height,
            round_to,
            round_mode,
        )

        width = min(width, MAX_DIMENSION)
        height = min(height, MAX_DIMENSION)

        aspect_ratio = float(width / height)

        info = (
            f"{family} | {width}x{height} | "
            f"latent {width // 8}x{height // 8} | "
            f"aspect {aspect_ratio:.4f}"
        )

        return (
            width,
            height,
            width // 8,
            height // 8,
            aspect_ratio,
            info,
        )

# ============================================================
# 2. LoRA Trigger Words
# ============================================================

class LocalLoRATriggerWords:
    @classmethod
    def INPUT_TYPES(cls):
        loras = [
            NONE_LORA
        ] + folder_paths.get_filename_list("loras")

        return {
            "required": {
                "lora_name": (
                    loras,
                    {
                        "default": NONE_LORA,
                    },
                ),
                "max_words": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                    },
                ),
                "separator": (
                    [
                        ", ",
                        "\n",
                        " ",
                    ],
                    {
                        "default": ", ",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )

    RETURN_NAMES = (
        "trigger_words",
        "info",
    )

    FUNCTION = "read_trigger_words"
    CATEGORY = "Local Tools/LoRA"

    def read_trigger_words(
        self,
        lora_name,
        max_words,
        separator,
    ):
        if (
            not lora_name
            or lora_name == NONE_LORA
        ):
            return (
                "",
                "No LoRA selected",
            )

        path = folder_paths.get_full_path_or_raise(
            "loras",
            lora_name,
        )

        metadata = read_lora_metadata(path)

        words = extract_trigger_words(
            metadata,
            max_words,
        )

        trigger_words = separator.join(words)

        if words:
            info = (
                f"{lora_name}\n"
                f"Found trigger words: {len(words)}"
            )

        else:
            info = (
                f"{lora_name}\n"
                "No trigger words found in metadata"
            )

        return trigger_words, info

# ============================================================
# 3. LoRA Stack 10
# ============================================================

class LocalLoRAStack10:
    @classmethod
    def INPUT_TYPES(cls):
        loras = [
            NONE_LORA
        ] + folder_paths.get_filename_list("loras")

        required = {
            "model": (
                "MODEL",
                {
                    "tooltip": (
                        "Base diffusion model."
                    ),
                },
            ),
            "clip": (
                "CLIP",
                {
                    "tooltip": (
                        "CLIP text encoder."
                    ),
                },
            ),
        }

        for index in range(1, 11):
            required[f"lora_{index}"] = (
                loras,
                {
                    "default": NONE_LORA,
                },
            )

            required[
                f"strength_model_{index}"
            ] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                },
            )

            required[
                f"strength_clip_{index}"
            ] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                },
            )

        return {
            "required": required,
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "STRING",
        "STRING",
    )

    RETURN_NAMES = (
        "MODEL",
        "CLIP",
        "active_loras",
        "trigger_words",
    )

    FUNCTION = "apply_loras"
    CATEGORY = "Local Tools/LoRA"

    def apply_loras(
        self,
        model,
        clip,
        **kwargs,
    ):
        report = []
        all_trigger_words = []

        for index in range(1, 11):
            lora_name = kwargs.get(
                f"lora_{index}",
                NONE_LORA,
            )

            strength_model = float(
                kwargs.get(
                    f"strength_model_{index}",
                    1.0,
                )
            )

            strength_clip = float(
                kwargs.get(
                    f"strength_clip_{index}",
                    1.0,
                )
            )

            if (
                not lora_name
                or lora_name == NONE_LORA
                or (
                    strength_model == 0.0
                    and strength_clip == 0.0
                )
            ):
                continue

            path = folder_paths.get_full_path_or_raise(
                "loras",
                lora_name,
            )

            try:
                lora_state, metadata = (
                    load_lora_state_dict(path)
                )

                model, clip = apply_lora(
                    model,
                    clip,
                    lora_state,
                    metadata,
                    strength_model,
                    strength_clip,
                )

                trigger_words = extract_trigger_words(
                    metadata,
                    max_words=32,
                )

                all_trigger_words.extend(
                    trigger_words
                )

                report.append(
                    f"[APPLIED] slot {index}: "
                    f"{lora_name} "
                    f"(model={strength_model:g}, "
                    f"clip={strength_clip:g})"
                )

            except Exception as error:
                raise RuntimeError(
                    "LoRA Stack 10 failed in "
                    f"slot {index}: {lora_name}\n"
                    f"{error}"
                ) from error

        all_trigger_words = deduplicate_words(
            all_trigger_words
        )

        if report:
            report_text = "\n".join(report)

        else:
            report_text = (
                "[LoRA Stack 10] "
                "No LoRAs applied"
            )

        trigger_text = ", ".join(
            all_trigger_words
        )

        return (
            model,
            clip,
            report_text,
            trigger_text,
        )

# ============================================================
# 4. Latent Upscale Exact Size
# ============================================================

class LocalLatentUpscaleExact:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": (
                    "LATENT",
                    {},
                ),
                "target_width": (
                    "INT",
                    {
                        "default": 640,
                        "min": 64,
                        "max": MAX_DIMENSION,
                        "step": 8,
                    },
                ),
                "target_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": MAX_DIMENSION,
                        "step": 8,
                    },
                ),
                "upscale_method": (
                    [
                        "nearest-exact",
                        "bilinear",
                        "area",
                        "bicubic",
                        "bislerp",
                        "lanczos",
                    ],
                    {
                        "default": "bislerp",
                    },
                ),
                "crop": (
                    [
                        "disabled",
                        "center",
                    ],
                    {
                        "default": "disabled",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "LATENT",
        "INT",
        "INT",
        "STRING",
    )

    RETURN_NAMES = (
        "samples",
        "width",
        "height",
        "info",
    )

    FUNCTION = "upscale_exact"
    CATEGORY = "Local Tools/Latent"

    def upscale_exact(
        self,
        samples,
        target_width,
        target_height,
        upscale_method,
        crop,
    ):
        if not isinstance(samples, dict):
            raise TypeError(
                "Expected LATENT dictionary."
            )

        latent = samples.get("samples")

        if latent is None:
            raise ValueError(
                "LATENT does not contain samples."
            )

        width = round_dimension(
            target_width,
            multiple=8,
            mode="nearest",
        )

        height = round_dimension(
            target_height,
            multiple=8,
            mode="nearest",
        )

        latent_width = width // 8
        latent_height = height // 8

        output = samples.copy()

        output["samples"] = (
            comfy.utils.common_upscale(
                latent,
                latent_width,
                latent_height,
                upscale_method,
                crop,
            )
        )

        current_height = latent.shape[-2]
        current_width = latent.shape[-1]

        info = (
            f"{current_width * 8}x"
            f"{current_height * 8} "
            f"-> {width}x{height} | "
            f"latent {current_width}x"
            f"{current_height} -> "
            f"{latent_width}x{latent_height} | "
            f"{upscale_method} | "
            f"crop={crop}"
        )

        return (
            output,
            width,
            height,
            info,
        )

# ============================================================
# 5. Prompt Preset
# ============================================================

PROMPT_PRESETS = {
    "Empty": (
        "",
        "",
    ),

    "SD 1.5 | anime illustration": (
        (
            "masterpiece, best quality, high quality, "
            "detailed anime illustration"
        ),
        (
            "lowres, worst quality, low quality, "
            "bad anatomy, bad hands, extra fingers, "
            "extra limbs, text, watermark"
        ),
    ),

    "SD 1.5 | semi-realistic portrait": (
        (
            "masterpiece, best quality, detailed "
            "semi-realistic portrait, beautiful face, "
            "detailed eyes, detailed hair"
        ),
        (
            "lowres, worst quality, low quality, "
            "bad anatomy, bad hands, deformed face, "
            "extra fingers, text, watermark"
        ),
    ),

    "SDXL | illustration": (
        (
            "high quality illustration, detailed, "
            "clean composition, refined lineart, "
            "beautiful lighting"
        ),
        (
            "low quality, blurry, bad anatomy, "
            "bad hands, extra fingers, extra limbs, "
            "text, watermark"
        ),
    ),

    "SDXL | portrait": (
        (
            "high quality portrait, detailed face, "
            "detailed eyes, natural skin, "
            "cinematic lighting"
        ),
        (
            "low quality, blurry, distorted face, "
            "bad anatomy, bad hands, extra fingers, "
            "text, watermark"
        ),
    ),
}

class LocalPromptPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    list(PROMPT_PRESETS.keys()),
                    {
                        "default": "Empty",
                    },
                ),
                "trigger_words": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "extra_positive": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "extra_negative": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "separator": (
                    [
                        ", ",
                        "\n",
                    ],
                    {
                        "default": ", ",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )

    RETURN_NAMES = (
        "positive",
        "negative",
        "info",
    )

    FUNCTION = "make_prompt"
    CATEGORY = "Local Tools/Prompt"

    def make_prompt(
        self,
        preset,
        trigger_words,
        extra_positive,
        extra_negative,
        separator,
    ):
        base_positive, base_negative = (
            PROMPT_PRESETS[preset]
        )

        positive = join_prompt_parts(
            [
                base_positive,
                trigger_words,
                extra_positive,
            ],
            separator,
        )

        negative = join_prompt_parts(
            [
                base_negative,
                extra_negative,
            ],
            separator,
        )

        return (
            positive,
            negative,
            f"Prompt preset: {preset}",
        )

# ============================================================
# 6. Prompt Combine
# ============================================================

class LocalPromptCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "trigger_words": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "style": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "quality": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "composition": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "details": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "negative": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "separator": (
                    [
                        ", ",
                        "\n",
                    ],
                    {
                        "default": ", ",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )

    RETURN_NAMES = (
        "positive",
        "negative",
        "info",
    )

    FUNCTION = "combine_prompt"
    CATEGORY = "Local Tools/Prompt"

    def combine_prompt(
        self,
        subject,
        trigger_words,
        style,
        quality,
        composition,
        details,
        negative,
        separator,
    ):
        positive = join_prompt_parts(
            [
                subject,
                trigger_words,
                style,
                quality,
                composition,
                details,
            ],
            separator,
        )

        negative = clean_text(negative)

        part_count = sum(
            bool(clean_text(value))
            for value in [
                subject,
                trigger_words,
                style,
                quality,
                composition,
                details,
            ]
        )

        info = (
            f"positive parts: {part_count} | "
            f"negative: "
            f"{'yes' if negative else 'no'}"
        )

        return (
            positive,
            negative,
            info,
        )

# ============================================================
# 7. Checkpoint Loader — No VAE
# ============================================================

class LocalCheckpointLoaderNoVAE:
    """
    Loads MODEL and CLIP, but does not create the VAE
    embedded inside the checkpoint.

    The VAE output is retained only for workflow compatibility
    and intentionally returns None.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list(
                        "checkpoints"
                    ),
                    {},
                ),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "VAE",
        "STRING",
    )

    RETURN_NAMES = (
        "MODEL",
        "CLIP",
        "VAE",
        "info",
    )

    FUNCTION = "load_checkpoint"
    CATEGORY = "Local Tools/Loaders"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = (
            folder_paths.get_full_path_or_raise(
                "checkpoints",
                ckpt_name,
            )
        )

        result = (
            comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=False,
                output_clip=True,
                output_clipvision=False,
                output_model=True,
                embedding_directory=(
                    folder_paths.get_folder_paths(
                        "embeddings"
                    )
                ),
            )
        )

        model = result[0]
        clip = result[1]

        if model is None:
            raise RuntimeError(
                "Checkpoint did not produce MODEL."
            )

        if clip is None:
            raise RuntimeError(
                "Checkpoint did not produce CLIP."
            )

        info = (
            "Loaded MODEL + CLIP without "
            f"embedded VAE:\n{ckpt_name}"
        )

        return (
            model,
            clip,
            None,
            info,
        )

# ============================================================
# 8. Fast TAESD Decode Batched V1
# ============================================================

class LocalFastTAESDDecodeBatchedV1Legacy:
    """
    Fast tiled TAESD decoder.

    Uses only the lightweight TAESD decoder from
    models/vae_approx.

    It does not use a full checkpoint VAE.
    """

    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": (
                    "LATENT",
                    {},
                ),
                "taesd_model": (
                    [
                        "taesd",
                        "taesdxl",
                    ],
                    {
                        "default": "taesdxl",
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 2048,
                        "step": 8,
                    },
                ),
                "batch_tiles": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                    },
                ),
                "precision": (
                    [
                        "fp16",
                        "fp32",
                    ],
                    {
                        "default": "fp16",
                    },
                ),
                "output_device": (
                    [
                        "gpu",
                        "cpu",
                    ],
                    {
                        "default": "gpu",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
    )

    RETURN_NAMES = (
        "image",
    )

    FUNCTION = "decode"
    CATEGORY = "Local Tools/TAESD"

    DESCRIPTION = (
        "Fast tiled TAESD decoder for SD 1.5 and SDXL."
    )

    @classmethod
    def find_decoder(cls, taesd_model):
        files = folder_paths.get_filename_list(
            "vae_approx"
        )

        prefix = f"{taesd_model}_decoder."

        candidates = [
            filename
            for filename in files
            if filename.startswith(prefix)
        ]

        if not candidates:
            raise FileNotFoundError(
                f"TAESD decoder not found for "
                f"'{taesd_model}'.\n\n"
                "Expected one of:\n"
                f"models/vae_approx/"
                f"{taesd_model}_decoder.pth\n"
                f"models/vae_approx/"
                f"{taesd_model}_decoder.safetensors"
            )

        return folder_paths.get_full_path_or_raise(
            "vae_approx",
            candidates[0],
        )

    @classmethod
    def get_decoder(
        cls,
        taesd_model,
        latent_channels,
        precision,
        output_device,
    ):
        if latent_channels != 4:
            raise ValueError(
                "This node supports 4-channel "
                "SD 1.5/SDXL latents only."
            )

        decoder_path = cls.find_decoder(
            taesd_model
        )

        if output_device == "gpu":
            device = (
                comfy.model_management
                .get_torch_device()
            )

        else:
            device = torch.device("cpu")

        if isinstance(device, str):
            device = torch.device(device)

        if (
            precision == "fp16"
            and device.type != "cpu"
        ):
            dtype = torch.float16

        else:
            dtype = torch.float32

        cache_key = (
            decoder_path,
            taesd_model,
            latent_channels,
            str(device),
            str(dtype),
        )

        if cache_key not in cls._model_cache:
            decoder = TAESD(
                encoder_path=None,
                decoder_path=decoder_path,
                latent_channels=latent_channels,
            )

            decoder.eval()
            decoder.to(
                device=device,
                dtype=dtype,
            )

            # Те же scale values, которые использует
            # официальный загрузчик TAESD ComfyUI.
            if taesd_model == "taesd":
                scale = 0.18215

            elif taesd_model == "taesdxl":
                scale = 0.13025

            else:
                scale = 1.0

            with torch.no_grad():
                decoder.vae_scale.data.fill_(
                    scale
                )
                decoder.vae_shift.data.zero_()

            cls._model_cache[cache_key] = (
                decoder,
                device,
                dtype,
            )

        return cls._model_cache[cache_key]

    @staticmethod
    def safe_tile_values(
        tile_size,
        overlap,
    ):
        # TAESD декодирует latent с коэффициентом 8.
        tile_latent = max(
            8,
            int(tile_size) // 8,
        )

        overlap_latent = max(
            0,
            int(overlap) // 8,
        )

        if overlap_latent >= tile_latent:
            overlap_latent = max(
                0,
                tile_latent // 2,
            )

        return (
            tile_latent,
            overlap_latent,
        )

    def decode_chunk(
        self,
        decoder,
        latent_chunk,
        tile_latent,
        overlap_latent,
        device,
        output_device,
        dtype,
    ):
        latent_chunk = latent_chunk.contiguous()

        def decode_tile(tile):
            tile = tile.to(
                device=device,
                dtype=dtype,
            )

            decoded = decoder.decode(tile)

            return decoded.to(
                dtype=torch.float32
            )

        decoded = comfy.utils.tiled_scale(
            latent_chunk,
            decode_tile,
            tile_x=tile_latent,
            tile_y=tile_latent,
            overlap=overlap_latent,
            upscale_amount=8,
            out_channels=3,
            output_device=output_device,
        )

        return decoded

    def decode(
        self,
        samples,
        taesd_model,
        tile_size,
        overlap,
        batch_tiles,
        precision,
        output_device,
    ):
        if not isinstance(samples, dict):
            raise TypeError(
                "Fast TAESD Decode expects LATENT."
            )

        latent = samples.get("samples")

        if latent is None:
            raise ValueError(
                "LATENT does not contain samples."
            )

        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]

        # Стандартные image-latents имеют [B,C,H,W].
        # Для 5D берём первый temporal/frame dimension.
        if latent.ndim == 5:
            latent = latent[:, :, 0]

        if latent.ndim != 4:
            raise ValueError(
                "Expected latent shape [B,C,H,W], "
                f"got {tuple(latent.shape)}"
            )

        latent_channels = int(
            latent.shape[1]
        )

        decoder, device, dtype = (
            self.get_decoder(
                taesd_model=taesd_model,
                latent_channels=latent_channels,
                precision=precision,
                output_device=output_device,
            )
        )

        if output_device == "gpu":
            output_target = device

        else:
            output_target = torch.device("cpu")

        tile_latent, overlap_latent = (
            self.safe_tile_values(
                tile_size,
                overlap,
            )
        )

        chunk_size = max(
            1,
            int(batch_tiles),
        )

        decoded_parts = []

        with torch.inference_mode():
            for start in range(
                0,
                latent.shape[0],
                chunk_size,
            ):
                end = start + chunk_size

                latent_chunk = latent[start:end].to(
                    device=device,
                    dtype=dtype,
                )

                decoded = self.decode_chunk(
                    decoder=decoder,
                    latent_chunk=latent_chunk,
                    tile_latent=tile_latent,
                    overlap_latent=overlap_latent,
                    device=device,
                    output_device=output_target,
                    dtype=dtype,
                )

                decoded_parts.append(decoded)

                del latent_chunk
                del decoded

        # Снова собираем output после удаления
        # временных ссылок на chunk.
        decoded_parts = [
            part
            for part in decoded_parts
            if part is not None
        ]

        if not decoded_parts:
            raise RuntimeError(
                "TAESD produced no output."
            )

        image = torch.cat(
            decoded_parts,
            dim=0,
        )

        # TAESD отдаёт примерно [-1, 1].
        image = (
            (image.float() + 1.0) / 2.0
        ).clamp(
            0.0,
            1.0,
        )

        # ComfyUI IMAGE format: [B,H,W,C].
        image = image.movedim(
            1,
            -1,
        ).contiguous()

        return (
            image,
        )

# ============================================================
# Пути
# ============================================================

NODE_FOLDER = Path(__file__).resolve().parent
PROMPT_LISTS_FOLDER = NODE_FOLDER / "prompt_lists"

NONE_LIST = "[None]"
NONE_MODE = "disabled"

# ============================================================
# Загрузка старых нод
# ============================================================

def load_legacy_nodes():
    """
    Загружает предыдущую версию __init__.py из legacy_nodes.py,
    чтобы уже установленные ноды не исчезли.
    """

    legacy_path = NODE_FOLDER / "legacy_nodes.py"

    if not legacy_path.exists():
        return {}, {}

    try:
        module_name = (
            "ComfyUI_mini_pack_nodes_legacy"
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            str(legacy_path),
        )

        if spec is None or spec.loader is None:
            print(
                "[mini-pack-nodes] "
                "Cannot load legacy_nodes.py"
            )
            return {}, {}

        legacy_module = (
            importlib.util.module_from_spec(spec)
        )

        spec.loader.exec_module(legacy_module)

        old_class_mappings = getattr(
            legacy_module,
            "NODE_CLASS_MAPPINGS",
            {},
        )

        old_display_mappings = getattr(
            legacy_module,
            "NODE_DISPLAY_NAME_MAPPINGS",
            {},
        )

        print(
            "[mini-pack-nodes] "
            f"Loaded legacy nodes: "
            f"{len(old_class_mappings)}"
        )

        return (
            dict(old_class_mappings),
            dict(old_display_mappings),
        )

    except Exception as error:
        print(
            "[mini-pack-nodes] "
            f"Legacy nodes were not loaded: {error}"
        )

        return {}, {}

LEGACY_NODE_CLASS_MAPPINGS, \
LEGACY_NODE_DISPLAY_NAME_MAPPINGS = (
    load_legacy_nodes()
)

# ============================================================
# Общие функции для TXT-списков
# ============================================================

def ensure_prompt_lists_folder():
    PROMPT_LISTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

def available_prompt_lists():
    """
    Возвращает список TXT-файлов внутри prompt_lists.
    """

    ensure_prompt_lists_folder()

    files = []

    for path in PROMPT_LISTS_FOLDER.rglob("*.txt"):
        if path.is_file():
            relative = path.relative_to(
                PROMPT_LISTS_FOLDER
            )

            files.append(
                relative.as_posix()
            )

    files.sort(
        key=lambda item: item.casefold()
    )

    return files

def read_prompt_list(filename):
    """
    Каждая непустая строка TXT-файла —
    отдельный вариант для случайного выбора.

    Строки, начинающиеся с #, считаются комментариями.
    """

    if not filename or filename == NONE_LIST:
        return []

    path = (
        PROMPT_LISTS_FOLDER / filename
    ).resolve()

    try:
        path.relative_to(
            PROMPT_LISTS_FOLDER.resolve()
        )

    except ValueError:
        raise ValueError(
            "Invalid prompt list path."
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt list not found: {path}"
        )

    result = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            result.append(line)

    return result

def clean_prompt_part(value):
    if value is None:
        return ""

    return str(value).strip()

def combine_prompt_parts(
    parts,
    separator="\n\n",
):
    cleaned = []

    for part in parts:
        part = clean_prompt_part(part)

        if part:
            cleaned.append(part)

    return separator.join(cleaned)

# ============================================================
# Prompt Randomizer 10 — TXT
# ============================================================

class LocalPromptRandomizer10TXT:
    """
    Выбирает по одной строке из десяти TXT-списков.

    Для каждого слота доступны режимы:

    disabled:
        слот выключен;

    random:
        случайная строка из TXT;

    fixed:
        конкретная строка по номеру.

    randomize_each_queue=True:
        новый случайный набор при каждом запуске.

    randomize_each_queue=False:
        результат зависит от seed и воспроизводим.
    """

    @classmethod
    def INPUT_TYPES(cls):
        ensure_prompt_lists_folder()

        list_choices = [
            NONE_LIST
        ] + available_prompt_lists()

        required = {
            "seed": (
                "INT",
                {
                    "default": 123456789,
                    "min": 0,
                    "max": 0x7FFFFFFFFFFFFFFF,
                    "step": 1,
                },
            ),
            "randomize_each_queue": (
                "BOOLEAN",
                {
                    "default": True,
                },
            ),
            "extra_prompt": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                },
            ),
        }

        for index in range(1, 11):
            required[
                f"list_{index}"
            ] = (
                list_choices,
                {
                    "default": NONE_LIST,
                },
            )

            required[
                f"mode_{index}"
            ] = (
                [
                    "disabled",
                    "random",
                    "fixed",
                ],
                {
                    "default": "disabled",
                },
            )

            required[
                f"line_{index}"
            ] = (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                },
            )

        return {
            "required": required,
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )

    RETURN_NAMES = (
        "prompt",
        "info",
    )

    FUNCTION = "generate_prompt"

    CATEGORY = "Local Tools/Prompt"

    DESCRIPTION = (
        "Chooses prompt fragments from up to "
        "10 TXT lists."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Если включён randomize_each_queue,
        ComfyUI не должен возвращать старый cached output.
        """

        if kwargs.get(
            "randomize_each_queue",
            True,
        ):
            return float("nan")

        return ""

    def generate_prompt(
        self,
        seed,
        randomize_each_queue,
        extra_prompt,
        **kwargs,
    ):
        ensure_prompt_lists_folder()

        if randomize_each_queue:
            chooser = random.SystemRandom()

        else:
            chooser = random.Random(
                int(seed)
            )

        selected = []
        info_lines = []

        extra_prompt = clean_prompt_part(
            extra_prompt
        )

        if extra_prompt:
            selected.append(extra_prompt)

            info_lines.append(
                "[extra_prompt] "
                + extra_prompt
            )

        for index in range(1, 11):
            filename = kwargs.get(
                f"list_{index}",
                NONE_LIST,
            )

            mode = kwargs.get(
                f"mode_{index}",
                "disabled",
            )

            line_number = int(
                kwargs.get(
                    f"line_{index}",
                    1,
                )
            )

            if (
                mode == "disabled"
                or not filename
                or filename == NONE_LIST
            ):
                continue

            lines = read_prompt_list(
                filename
            )

            if not lines:
                info_lines.append(
                    f"[slot {index}] "
                    f"{filename}: empty"
                )

                continue

            if mode == "random":
                selected_line = chooser.choice(
                    lines
                )

                selected_number = (
                    lines.index(selected_line) + 1
                )

            elif mode == "fixed":
                selected_number = max(
                    1,
                    min(
                        line_number,
                        len(lines),
                    ),
                )

                selected_line = (
                    lines[selected_number - 1]
                )

            else:
                continue

            selected.append(selected_line)

            info_lines.append(
                f"[slot {index}] "
                f"{filename} | "
                f"{mode} | "
                f"line {selected_number}: "
                f"{selected_line}"
            )

        prompt = combine_prompt_parts(
            selected,
            separator=", ",
        )

        if not prompt:
            prompt = ""

        info = (
            f"seed={seed} | "
            f"randomize_each_queue="
            f"{randomize_each_queue}\n"
        )

        if info_lines:
            info += "\n".join(info_lines)

        else:
            info += "No prompt fragments selected."

        return (
            prompt,
            info,
        )

# ============================================================
# CLIP Text Encode 2-Part
# ============================================================

class LocalCLIPTextEncode2Part:
    """
    Разделённый CLIP Text Encode.

    quality_prompt:
        постоянные параметры качества и стиля;

    character_prompt:
        описание персонажа;

    random_prompt:
        выход Prompt Randomizer 10 — TXT.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {},
                ),
                "quality_prompt": (
                    "STRING",
                    {
                        "default": (
                            "masterpiece:1.2, "
                            "high quality, "
                            "best quality, "
                            "aesthetic"
                        ),
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
                "character_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
            },
            "optional": {
                "random_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "dynamicPrompts": True,
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )

    RETURN_NAMES = (
        "conditioning",
        "combined_prompt",
    )

    FUNCTION = "encode"

    CATEGORY = "Local Tools/Conditioning"

    DESCRIPTION = (
        "Encodes quality, character and "
        "random prompt separately."
    )

    def encode(
        self,
        clip,
        quality_prompt,
        character_prompt,
        random_prompt="",
    ):
        if clip is None:
            raise RuntimeError(
                "CLIP input is invalid."
            )

        combined_prompt = (
            combine_prompt_parts(
                [
                    quality_prompt,
                    character_prompt,
                    random_prompt,
                ],
                separator="\n\n",
            )
        )

        tokens = clip.tokenize(
            combined_prompt
        )

        conditioning = (
            clip.encode_from_tokens_scheduled(
                tokens
            )
        )

        return (
            conditioning,
            combined_prompt,
        )

# ============================================================
# Fast TAESD Decode Batched V1
# ============================================================

class LocalFastTAESDDecodeBatchedV1:
    """
    Быстрый тайловый TAESD decoder.

    Поддерживает:
        taesd   — SD 1.5;
        taesdxl — SDXL.

    Использует только decoder из:
        models/vae_approx
    """

    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": (
                    "LATENT",
                    {},
                ),
                "taesd_model": (
                    [
                        "taesd",
                        "taesdxl",
                    ],
                    {
                        "default": "taesd",
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 2048,
                        "step": 8,
                    },
                ),
                "batch_tiles": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                    },
                ),
                "precision": (
                    [
                        "fp16",
                        "fp32",
                    ],
                    {
                        "default": "fp16",
                    },
                ),
                "output_device": (
                    [
                        "gpu",
                        "cpu",
                    ],
                    {
                        "default": "gpu",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
    )

    RETURN_NAMES = (
        "image",
    )

    FUNCTION = "decode"

    CATEGORY = "Local Tools/TAESD"

    DESCRIPTION = (
        "Fast tiled TAESD decoder."
    )

    @classmethod
    def find_decoder(cls, taesd_model):
        files = folder_paths.get_filename_list(
            "vae_approx"
        )

        prefix = (
            f"{taesd_model}_decoder."
        )

        candidates = [
            filename
            for filename in files
            if filename.startswith(prefix)
        ]

        if not candidates:
            raise FileNotFoundError(
                "TAESD decoder not found.\n\n"
                f"Expected:\n"
                f"models/vae_approx/"
                f"{taesd_model}_decoder.pth\n"
                f"or:\n"
                f"models/vae_approx/"
                f"{taesd_model}_decoder.safetensors"
            )

        return folder_paths.get_full_path_or_raise(
            "vae_approx",
            candidates[0],
        )

    @classmethod
    def get_decoder(
        cls,
        taesd_model,
        precision,
        output_device,
    ):
        decoder_path = cls.find_decoder(
            taesd_model
        )

        device = (
            comfy.model_management
            .get_torch_device()
            if output_device == "gpu"
            else torch.device("cpu")
        )

        if isinstance(device, str):
            device = torch.device(device)

        if (
            precision == "fp16"
            and device.type != "cpu"
        ):
            dtype = torch.float16

        else:
            dtype = torch.float32

        cache_key = (
            decoder_path,
            taesd_model,
            str(device),
            str(dtype),
        )

        if cache_key not in cls._cache:
            decoder = TAESD(
                encoder_path=None,
                decoder_path=decoder_path,
                latent_channels=4,
            )

            decoder.eval()

            decoder.to(
                device=device,
                dtype=dtype,
            )

            if taesd_model == "taesd":
                scale = 0.18215
                shift = 0.0

            elif taesd_model == "taesdxl":
                scale = 0.13025
                shift = 0.0

            else:
                scale = 1.0
                shift = 0.0

            with torch.no_grad():
                decoder.vae_scale.data.fill_(
                    scale
                )

                decoder.vae_shift.data.fill_(
                    shift
                )

            cls._cache[cache_key] = (
                decoder,
                device,
                dtype,
            )

        return cls._cache[cache_key]

    @staticmethod
    def tile_parameters(
        tile_size,
        overlap,
    ):
        # TAESD декодирует latent с увеличением 8x.
        tile_latent = max(
            8,
            int(tile_size) // 8,
        )

        overlap_latent = max(
            0,
            int(overlap) // 8,
        )

        if overlap_latent >= tile_latent:
            overlap_latent = max(
                0,
                tile_latent // 2,
            )

        return (
            tile_latent,
            overlap_latent,
        )

    @staticmethod
    def convert_latent_shape(latent):
        if latent.ndim == 5:
            # Для видео/temporal latent.
            latent = latent[:, :, 0]

        if latent.ndim != 4:
            raise ValueError(
                "Expected latent shape "
                "[B,C,H,W], got "
                f"{tuple(latent.shape)}"
            )

        return latent

    def decode_part(
        self,
        decoder,
        latent_part,
        tile_latent,
        overlap_latent,
        device,
        dtype,
        output_device,
    ):
        latent_part = latent_part.to(
            device=device,
            dtype=dtype,
        ).contiguous()

        def decode_tile(tile):
            tile = tile.to(
                device=device,
                dtype=dtype,
            ).contiguous()

            result = decoder.decode(tile)

            return result.float()

        return comfy.utils.tiled_scale(
            latent_part,
            decode_tile,
            tile_x=tile_latent,
            tile_y=tile_latent,
            overlap=overlap_latent,
            upscale_amount=8,
            out_channels=3,
            output_device=output_device,
        )

    def decode(
        self,
        samples,
        taesd_model,
        tile_size,
        overlap,
        batch_tiles,
        precision,
        output_device,
    ):
        if not isinstance(samples, dict):
            raise TypeError(
                "Fast TAESD expects LATENT."
            )

        latent = samples.get("samples")

        if latent is None:
            raise ValueError(
                "LATENT does not contain samples."
            )

        latent = self.convert_latent_shape(
            latent
        )

        decoder, device, dtype = (
            self.get_decoder(
                taesd_model=taesd_model,
                precision=precision,
                output_device=output_device,
            )
        )

        if output_device == "gpu":
            tiled_output_device = device

        else:
            tiled_output_device = torch.device(
                "cpu"
            )

        tile_latent, overlap_latent = (
            self.tile_parameters(
                tile_size,
                overlap,
            )
        )

        chunk_size = max(
            1,
            int(batch_tiles),
        )

        parts = []

        with torch.inference_mode():
            for start in range(
                0,
                latent.shape[0],
                chunk_size,
            ):
                end = start + chunk_size

                latent_part = latent[start:end]

                decoded = self.decode_part(
                    decoder=decoder,
                    latent_part=latent_part,
                    tile_latent=tile_latent,
                    overlap_latent=overlap_latent,
                    device=device,
                    dtype=dtype,
                    output_device=tiled_output_device,
                )

                parts.append(decoded)

        if not parts:
            raise RuntimeError(
                "TAESD returned no image."
            )

        image = torch.cat(
            parts,
            dim=0,
        )

        # TAESD output is approximately [-1, 1].
        image = (
            (image.float() + 1.0) / 2.0
        ).clamp(
            0.0,
            1.0,
        )

        # ComfyUI IMAGE format: B,H,W,C.
        image = image.movedim(
            1,
            -1,
        ).contiguous()

        return (
            image,
        )

# ============================================================
# Registration
# ============================================================

NODE_CLASS_MAPPINGS = dict(
    LEGACY_NODE_CLASS_MAPPINGS
)

NODE_DISPLAY_NAME_MAPPINGS = dict(
    LEGACY_NODE_DISPLAY_NAME_MAPPINGS
)

NODE_CLASS_MAPPINGS.update(
    {
        "LocalResolutionPreset": (
            LocalResolutionPreset
        ),
        "LocalLoRATriggerWords": (
            LocalLoRATriggerWords
        ),
        "LocalLoRAStack10": (
            LocalLoRAStack10
        ),
        "LocalLatentUpscaleExact": (
            LocalLatentUpscaleExact
        ),
        "LocalPromptPreset": (
            LocalPromptPreset
        ),
        "LocalPromptCombine": (
            LocalPromptCombine
        ),
        "LocalCheckpointLoaderNoVAE": (
            LocalCheckpointLoaderNoVAE
        ),
        "LocalFastTAESDDecodeBatchedV1Legacy": (
            LocalFastTAESDDecodeBatchedV1Legacy
        ),
        "LocalPromptRandomizer10TXT": (
            LocalPromptRandomizer10TXT
        ),
        "LocalCLIPTextEncode2Part": (
            LocalCLIPTextEncode2Part
        ),
        "LocalFastTAESDDecodeBatchedV1": (
            LocalFastTAESDDecodeBatchedV1
        ),
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "LocalResolutionPreset": (
            "Resolution Preset SDXL / SD1.5"
        ),
        "LocalLoRATriggerWords": (
            "LoRA Trigger Words"
        ),
        "LocalLoRAStack10": (
            "LoRA Stack 10"
        ),
        "LocalLatentUpscaleExact": (
            "Latent Upscale Exact Size"
        ),
        "LocalPromptPreset": (
            "Prompt Preset"
        ),
        "LocalPromptCombine": (
            "Prompt Combine"
        ),
        "LocalCheckpointLoaderNoVAE": (
            "Checkpoint Loader - No VAE"
        ),
        "LocalFastTAESDDecodeBatchedV1Legacy": (
            "Fast TAESD Decode Batched V1 - Legacy"
        ),
        "LocalPromptRandomizer10TXT": (
            "Prompt Randomizer 10 - TXT"
        ),
        "LocalCLIPTextEncode2Part": (
            "CLIP Text Encode 2-Part"
        ),
        "LocalFastTAESDDecodeBatchedV1": (
            "Fast TAESD Decode Batched V1"
        ),
    }
)

print(
    "[mini-pack-nodes] "
    "Prompt Randomizer 10 - TXT loaded"
)

print(
    "[mini-pack-nodes] "
    "CLIP Text Encode 2-Part loaded"
)
