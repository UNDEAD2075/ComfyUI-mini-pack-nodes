# ComfyUI-mini-pack-nodes

A compact collection of utility nodes for ComfyUI workflows.

The pack focuses on practical workflow helpers for:

- resolution management;
- LoRA metadata and trigger words;
- applying up to ten LoRAs;
- exact latent upscaling;
- prompt presets;
- structured prompt composition;
- checkpoint loading without an embedded VAE;
- prompt randomization from TXT files;
- two-part CLIP conditioning;
- fast tiled TAESD decoding.

The nodes are designed for modular and reusable ComfyUI workflows.

---

## Features

- Resolution presets for SD 1.5 and SDXL.
- Custom resolution support.
- Automatic dimension rounding.
- LoRA trigger-word extraction from metadata.
- Support for common Kohya and ModelSpec metadata formats.
- Ten-slot LoRA stack.
- Separate model and CLIP LoRA strengths.
- Exact latent resizing to a requested output size.
- Prompt presets for common SD 1.5 and SDXL workflows.
- Structured positive and negative prompt composition.
- Checkpoint loading without creating the checkpoint VAE.
- Prompt randomization from up to ten TXT lists.
- Seed-based reproducible prompt selection.
- True random prompt selection for every queue.
- Separate quality, character and random prompt inputs.
- Fast tiled TAESD decoding for SD 1.5 and SDXL.
- Batch processing of latent samples.
- GPU or CPU TAESD output.
- FP16 and FP32 TAESD decoding.
- Optional legacy-node loading.

---

## Installation

Clone the repository into the `custom_nodes` directory of ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/UNDEAD2075/ComfyUI-mini-pack-nodes.git
```

Restart ComfyUI after installation.

For an existing installation:

```bash
cd ComfyUI/custom_nodes/ComfyUI-mini-pack-nodes
git pull
```

---

## Important Current-Code Fix

The current implementation uses `Path`, `random` and `importlib.util`.

Make sure the following imports exist at the top of `__init__.py`:

```python
import importlib.util
import random
from pathlib import Path
```

Without these imports, the node pack may fail during startup.

---

## Node Categories

The nodes are registered under the following categories:

- `Local Tools/Resolution`
- `Local Tools/LoRA`
- `Local Tools/Latent`
- `Local Tools/Prompt`
- `Local Tools/Loaders`
- `Local Tools/Conditioning`
- `Local Tools/TAESD`

---

# Nodes

## 1. Resolution Preset

### Internal name

```text
LocalResolutionPreset
```

### Display name

```text
Resolution Preset SDXL / SD1.5
```

### Category

```text
Local Tools/Resolution
```

This node provides predefined image sizes for SD 1.5 and SDXL workflows.

### Inputs

- `preset`
- `custom_width`
- `custom_height`
- `round_to`
- `round_mode`

### Built-in presets

#### SDXL

- Portrait: `640x1024`
- Portrait: `768x1152`
- Portrait: `832x1216`
- Landscape: `1024x640`
- Landscape: `1152x768`
- Landscape: `1216x832`
- Square: `1024x1024`

#### SD 1.5

- Square: `512x512`
- Portrait: `512x768`
- Portrait: `576x864`
- Portrait: `640x960`
- Portrait: `640x1024`
- Landscape: `768x512`
- Landscape: `960x640`
- Landscape: `1024x640`

### Custom mode

When `preset` is set to `custom`, the node uses:

- `custom_width`;
- `custom_height`.

Dimensions are rounded to the selected multiple.

### Rounding modes

- `nearest`
- `floor`
- `ceil`

The node returns:

- image width;
- image height;
- latent width;
- latent height;
- aspect ratio;
- readable information string.

The maximum supported dimension is `8192`.

---

## 2. LoRA Trigger Words

### Internal name

```text
LocalLoRATriggerWords
```

### Display name

```text
LoRA Trigger Words
```

### Category

```text
Local Tools/LoRA
```

This node reads LoRA metadata without loading the complete LoRA into the model.

### Supported metadata keys

The node checks the following fields:

- `ss_trigger_words`
- `trigger_words`
- `trigger_word`
- `modelspec.trigger_phrase`
- `modelspec.trigger_words`
- `tags`

As a fallback, it can extract tags from:

```text
ss_tag_frequency
```

### Behavior

- Duplicate words are removed case-insensitively.
- Empty values are ignored.
- Lists, tuples, dictionaries and JSON strings are supported.
- Weighted dictionaries are sorted by weight.
- The number of returned words can be limited.

### Inputs

- LoRA file;
- maximum number of words;
- separator.

### Output

- trigger-word string;
- information string.

---

## 3. LoRA Stack 10

### Internal name

```text
LocalLoRAStack10
```

### Display name

```text
LoRA Stack 10
```

### Category

```text
Local Tools/LoRA
```

This node applies up to ten LoRAs in a single node.

### Inputs

- `MODEL`
- `CLIP`
- ten LoRA slots;
- ten model strength values;
- ten CLIP strength values.

Each slot contains:

- LoRA filename;
- model strength;
- CLIP strength.

The strength range is:

```text
-100.0 ... 100.0
```

### Behavior

A slot is skipped when:

- no LoRA is selected;
- the slot is disabled;
- both model and CLIP strengths are zero.

LoRAs are loaded using ComfyUI's official LoRA loading mechanism.

The node also collects trigger words from the applied LoRA metadata and removes duplicates.

### Outputs

- modified `MODEL`;
- modified `CLIP`;
- active LoRA report;
- combined trigger words.

---

## 4. Latent Upscale Exact Size

### Internal name

```text
LocalLatentUpscaleExact
```

### Display name

```text
Latent Upscale Exact Size
```

### Category

```text
Local Tools/Latent
```

This node resizes a LATENT object to the requested output resolution.

### Inputs

- `LATENT`;
- target width;
- target height;
- upscale method;
- crop mode.

### Supported methods

- `nearest-exact`
- `bilinear`
- `area`
- `bicubic`
- `bislerp`
- `lanczos`

### Crop modes

- `disabled`
- `center`

The requested dimensions are rounded to multiples of eight.

### Outputs

- resized `LATENT`;
- final width;
- final height;
- readable resize information.

---

## 5. Prompt Preset

### Internal name

```text
LocalPromptPreset
```

### Display name

```text
Prompt Preset
```

### Category

```text
Local Tools/Prompt
```

This node generates positive and negative prompts from a predefined template.

### Built-in presets

- `Empty`
- `SD 1.5 | anime illustration`
- `SD 1.5 | semi-realistic portrait`
- `SDXL | illustration`
- `SDXL | portrait`

### Inputs

- preset;
- trigger words;
- extra positive prompt;
- extra negative prompt;
- separator.

### Outputs

- positive prompt;
- negative prompt;
- information string.

Empty prompt parts are removed automatically.

---

## 6. Prompt Combine

### Internal name

```text
LocalPromptCombine
```

### Display name

```text
Prompt Combine
```

### Category

```text
Local Tools/Prompt
```

This node combines structured prompt sections.

### Positive sections

- subject;
- trigger words;
- style;
- quality;
- composition;
- details.

### Negative section

- negative prompt.

### Separators

- comma and space;
- newline.

Empty sections are skipped.

### Outputs

- combined positive prompt;
- cleaned negative prompt;
- information string.

---

## 7. Checkpoint Loader - No VAE

### Internal name

```text
LocalCheckpointLoaderNoVAE
```

### Display name

```text
Checkpoint Loader - No VAE
```

### Category

```text
Local Tools/Loaders
```

This node loads:

- `MODEL`;
- `CLIP`;

but intentionally does not create the VAE embedded in the checkpoint.

The VAE output is returned as:

```text
None
```

This is useful when the workflow uses:

- a separate VAE;
- TAESD;
- an external VAE loader;
- a custom decoder.

### Outputs

- `MODEL`;
- `CLIP`;
- `VAE` set to `None`;
- information string.

---

## 8. Fast TAESD Decode Batched V1

### Internal name

```text
LocalFastTAESDDecodeBatchedV1
```

### Display name

```text
Fast TAESD Decode Batched V1
```

### Category

```text
Local Tools/TAESD
```

This node decodes latent samples using the lightweight TAESD approximate decoder.

It does not use a full checkpoint VAE.

### Supported models

- `taesd` for SD 1.5;
- `taesdxl` for SDXL.

### Inputs

- `LATENT`;
- TAESD model;
- tile size;
- overlap;
- batch tile count;
- precision;
- output device.

### Precision

- `fp16`;
- `fp32`.

### Output device

- `gpu`;
- `cpu`.

### Tiled decoding

Large images are decoded tile by tile.

Important parameters:

- larger tile size can improve speed but requires more memory;
- smaller tile size reduces memory usage;
- overlap helps reduce tile seams;
- `batch_tiles` controls how many latent samples are decoded per batch.

### Decoder files

The node searches in:

```text
ComfyUI/models/vae_approx/
```

Expected filenames:

```text
taesd_decoder.pth
taesd_decoder.safetensors
taesdxl_decoder.pth
taesdxl_decoder.safetensors
```

The decoder is cached after loading.

The output is converted to the standard ComfyUI image layout:

```text
[B, H, W, C]
```

For five-dimensional latent input, the implementation uses the first temporal/frame dimension.

---

## 9. Fast TAESD Decode Batched V1 Legacy

### Internal name

```text
LocalFastTAESDDecodeBatchedV1Legacy
```

### Display name

```text
Fast TAESD Decode Batched V1 - Legacy
```

This is a compatibility version of the TAESD node.

It is kept to preserve workflows that used an earlier implementation.

---

## 10. Prompt Randomizer 10 - TXT

### Internal name

```text
LocalPromptRandomizer10TXT
```

### Display name

```text
Prompt Randomizer 10 - TXT
```

### Category

```text
Local Tools/Prompt
```

This node selects prompt fragments from up to ten TXT files.

### Prompt-list directory

The directory is created automatically:

```text
ComfyUI/custom_nodes/ComfyUI-mini-pack-nodes/prompt_lists/
```

Subdirectories are supported.

### TXT format

Each non-empty line is one prompt option.

Lines beginning with `#` are treated as comments.

Example:

```text
# Camera angles
close-up portrait
full body
cinematic side view
wide shot
```

### Slot modes

Each of the ten slots supports:

- `disabled`;
- `random`;
- `fixed`.

### Random mode

Selects one line randomly from the selected file.

### Fixed mode

Selects a specific line number.

If the requested number is outside the file range, it is clamped to the available range.

### Reproducibility

When `randomize_each_queue` is enabled:

- a new random selection is made for every queue;
- ComfyUI caching is bypassed.

When disabled:

- the result is deterministic;
- the selected result depends on the seed.

### Outputs

- generated prompt;
- detailed selection report.

The report contains:

- seed;
- randomization mode;
- selected file;
- selected mode;
- selected line number;
- selected text.

---

## 11. CLIP Text Encode 2-Part

### Internal name

```text
LocalCLIPTextEncode2Part
```

### Display name

```text
CLIP Text Encode 2-Part
```

### Category

```text
Local Tools/Conditioning
```

This node separates the prompt into logical sections before CLIP encoding.

### Inputs

- `CLIP`;
- quality prompt;
- character prompt;
- optional random prompt.

The optional random prompt can be connected directly to:

```text
Prompt Randomizer 10 - TXT
```

### Processing order

The sections are joined in this order:

1. quality prompt;
2. character prompt;
3. random prompt.

Sections are separated by two newline characters.

### Outputs

- `CONDITIONING`;
- combined prompt string.

---

# Recommended Workflow Examples

## Basic SDXL workflow

```text
Resolution Preset
    → Empty Latent Image

Prompt Preset
    → CLIP Text Encode

Checkpoint Loader - No VAE
    → MODEL
    → CLIP

Separate VAE Loader
    → VAE

KSampler
    → VAE Decode
```

## LoRA workflow

```text
Checkpoint Loader - No VAE
    → LoRA Stack 10
    → KSampler
```

Optional trigger-word flow:

```text
LoRA Trigger Words
    → Prompt Preset
    → CLIP Text Encode
```

## TXT prompt randomization

```text
Prompt Randomizer 10 - TXT
    → CLIP Text Encode 2-Part
    → KSampler
```

Recommended structure:

```text
Quality Prompt
Character Prompt
Random TXT Prompt
    → CLIP Text Encode 2-Part
```

## TAESD preview workflow

```text
KSampler
    → Fast TAESD Decode Batched V1
    → Preview Image
```

TAESD is intended for fast preview decoding. For final-quality output, use the full VAE when available.

---

# Limitations

- TAESD is an approximate decoder and is not a replacement for a full VAE.
- TAESD requires the appropriate decoder file in `models/vae_approx`.
- The package is currently concentrated in a single large `__init__.py`.
- The legacy loader expects an optional `legacy_nodes.py`.
- The current source requires additional standard-library imports mentioned above.
- LoRA metadata quality depends on the metadata written into the LoRA file.
- Prompt randomization only reads `.txt` files.
- The package does not automatically download models or decoder files.
- The nodes are intended for image-generation workflows and are not a complete video workflow pack.

---

# Русский README

# ComfyUI-mini-pack-nodes

Компактный набор вспомогательных нод для ComfyUI.

Пакет предназначен для создания удобных, модульных и переиспользуемых workflow:

- управление разрешением;
- preset-наборы для SD 1.5 и SDXL;
- чтение trigger words из LoRA;
- применение до десяти LoRA;
- точное масштабирование latent;
- готовые prompt-пресеты;
- структурированное объединение prompt;
- загрузка checkpoint без встроенного VAE;
- рандомизация prompt из TXT-файлов;
- разделённое CLIP-conditioning;
- быстрый тайловый TAESD decoder.

---

## Возможности

- Готовые разрешения SD 1.5 и SDXL.
- Пользовательское разрешение.
- Округление размеров до нужного значения.
- Чтение trigger words из metadata LoRA.
- Поддержка распространённых форматов Kohya и ModelSpec.
- Stack для десяти LoRA.
- Отдельная сила LoRA для MODEL и CLIP.
- Точное изменение размера LATENT.
- Prompt preset для SD 1.5 и SDXL.
- Раздельная сборка positive и negative prompt.
- Загрузка MODEL и CLIP без встроенного VAE.
- Рандомизация prompt из десяти TXT-списков.
- Воспроизводимый выбор через seed.
- Новый случайный выбор при каждом запуске queue.
- Разделение quality, character и random prompt.
- Быстрый tiled TAESD decoder.
- Batch-декодирование latent.
- Декодирование на GPU или CPU.
- Поддержка FP16 и FP32.
- Опциональная загрузка legacy-нод.

---

## Установка

Перейдите в папку `custom_nodes` вашего ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/UNDEAD2075/ComfyUI-mini-pack-nodes.git
```

Перезапустите ComfyUI.

Для обновления:

```bash
cd ComfyUI/custom_nodes/ComfyUI-mini-pack-nodes
git pull
```

---

## Важное исправление текущего кода

В `__init__.py` используются:

- `Path`;
- `random`;
- `importlib.util`.

В начало файла необходимо добавить:

```python
import importlib.util
import random
from pathlib import Path
```

Без этих импортов пакет может завершиться с ошибкой при запуске ComfyUI.

---

## Категории нод

Ноды находятся в следующих категориях:

- `Local Tools/Resolution`
- `Local Tools/LoRA`
- `Local Tools/Latent`
- `Local Tools/Prompt`
- `Local Tools/Loaders`
- `Local Tools/Conditioning`
- `Local Tools/TAESD`

---

# Ноды

## 1. Resolution Preset

### Внутреннее имя

```text
LocalResolutionPreset
```

### Отображаемое имя

```text
Resolution Preset SDXL / SD1.5
```

### Категория

```text
Local Tools/Resolution
```

Нода выдаёт готовые разрешения для SD 1.5 и SDXL.

### Входы

- `preset`;
- `custom_width`;
- `custom_height`;
- `round_to`;
- `round_mode`.

### Готовые пресеты SDXL

- Portrait: `640x1024`;
- Portrait: `768x1152`;
- Portrait: `832x1216`;
- Landscape: `1024x640`;
- Landscape: `1152x768`;
- Landscape: `1216x832`;
- Square: `1024x1024`.

### Готовые пресеты SD 1.5

- Square: `512x512`;
- Portrait: `512x768`;
- Portrait: `576x864`;
- Portrait: `640x960`;
- Portrait: `640x1024`;
- Landscape: `768x512`;
- Landscape: `960x640`;
- Landscape: `1024x640`.

### Пользовательский режим

При выборе `custom` используются:

- `custom_width`;
- `custom_height`.

Размеры округляются до выбранного множителя.

### Режимы округления

- `nearest`;
- `floor`;
- `ceil`.

### Выходы

- ширина изображения;
- высота изображения;
- ширина latent;
- высота latent;
- aspect ratio;
- строка с информацией.

Максимальный размер — `8192`.

---

## 2. LoRA Trigger Words

### Внутреннее имя

```text
LocalLoRATriggerWords
```

### Отображаемое имя

```text
LoRA Trigger Words
```

### Категория

```text
Local Tools/LoRA
```

Нода читает metadata LoRA, не загружая полностью веса LoRA в модель.

### Поддерживаемые ключи metadata

Проверяются следующие поля:

- `ss_trigger_words`;
- `trigger_words`;
- `trigger_word`;
- `modelspec.trigger_phrase`;
- `modelspec.trigger_words`;
- `tags`.

Дополнительно используется:

```text
ss_tag_frequency
```

### Обработка

- дубликаты удаляются без учёта регистра;
- пустые значения пропускаются;
- поддерживаются строки, списки, tuples, dictionaries и JSON;
- словари с числовыми весами сортируются по весу;
- количество слов можно ограничить.

### Входы

- LoRA;
- максимальное количество слов;
- разделитель.

### Выходы

- строка с trigger words;
- информационная строка.

---

## 3. LoRA Stack 10

### Внутреннее имя

```text
LocalLoRAStack10
```

### Отображаемое имя

```text
LoRA Stack 10
```

### Категория

```text
Local Tools/LoRA
```

Нода позволяет применить до десяти LoRA в одном месте.

### Входы

- `MODEL`;
- `CLIP`;
- десять LoRA-слотов;
- десять значений силы для MODEL;
- десять значений силы для CLIP.

Для каждого слота доступны:

- имя LoRA;
- `strength_model`;
- `strength_clip`.

Диапазон силы:

```text
-100.0 ... 100.0
```

### Поведение

Слот пропускается, если:

- LoRA не выбрана;
- слот отключён;
- обе силы равны нулю.

Для применения используется официальный механизм загрузки LoRA из ComfyUI.

Нода также собирает trigger words из metadata применённых LoRA и удаляет повторения.

### Выходы

- изменённый `MODEL`;
- изменённый `CLIP`;
- отчёт об активных LoRA;
- объединённые trigger words.

---

## 4. Latent Upscale Exact Size

### Внутреннее имя

```text
LocalLatentUpscaleExact
```

### Отображаемое имя

```text
Latent Upscale Exact Size
```

### Категория

```text
Local Tools/Latent
```

Нода изменяет размер объекта `LATENT` до указанного разрешения.

### Входы

- `LATENT`;
- целевая ширина;
- целевая высота;
- метод масштабирования;
- режим crop.

### Методы масштабирования

- `nearest-exact`;
- `bilinear`;
- `area`;
- `bicubic`;
- `bislerp`;
- `lanczos`.

### Режимы crop

- `disabled`;
- `center`.

Размеры округляются до значений, кратных восьми.

### Выходы

- изменённый `LATENT`;
- итоговая ширина;
- итоговая высота;
- строка с информацией.

---

## 5. Prompt Preset

### Внутреннее имя

```text
LocalPromptPreset
```

### Отображаемое имя

```text
Prompt Preset
```

### Категория

```text
Local Tools/Prompt
```

Нода создаёт positive и negative prompt на основе preset.

### Доступные preset

- `Empty`;
- `SD 1.5 | anime illustration`;
- `SD 1.5 | semi-realistic portrait`;
- `SDXL | illustration`;
- `SDXL | portrait`.

### Входы

- preset;
- trigger words;
- дополнительный positive prompt;
- дополнительный negative prompt;
- разделитель.

### Выходы

- positive prompt;
- negative prompt;
- информационная строка.

Пустые части автоматически удаляются.

---

## 6. Prompt Combine

### Внутреннее имя

```text
LocalPromptCombine
```

### Отображаемое имя

```text
Prompt Combine
```

### Категория

```text
Local Tools/Prompt
```

Нода объединяет prompt по смысловым секциям.

### Positive-секции

- subject;
- trigger words;
- style;
- quality;
- composition;
- details.

### Negative-секция

- negative.

### Разделители

- `, `;
- перевод строки.

Пустые секции пропускаются.

### Выходы

- объединённый positive prompt;
- очищенный negative prompt;
- информационная строка.

---

## 7. Checkpoint Loader - No VAE

### Внутреннее имя

```text
LocalCheckpointLoaderNoVAE
```

### Отображаемое имя

```text
Checkpoint Loader - No VAE
```

### Категория

```text
Local Tools/Loaders
```

Нода загружает:

- `MODEL`;
- `CLIP`;

но не создаёт VAE, встроенный в checkpoint.

На выходе `VAE` намеренно равен:

```text
None
```

Это удобно, когда используется:

- отдельный VAE;
- TAESD;
- внешний VAE loader;
- собственный decoder.

### Выходы

- `MODEL`;
- `CLIP`;
- `VAE = None`;
- информационная строка.

---

## 8. Fast TAESD Decode Batched V1

### Внутреннее имя

```text
LocalFastTAESDDecodeBatchedV1
```

### Отображаемое имя

```text
Fast TAESD Decode Batched V1
```

### Категория

```text
Local Tools/TAESD
```

Нода декодирует latent с помощью облегчённого approximate TAESD decoder.

Полный VAE checkpoint при этом не используется.

### Поддерживаемые модели

- `taesd` — SD 1.5;
- `taesdxl` — SDXL.

### Входы

- `LATENT`;
- модель TAESD;
- размер тайла;
- overlap;
- количество batch tiles;
- precision;
- устройство вывода.

### Precision

- `fp16`;
- `fp32`.

### Устройство вывода

- `gpu`;
- `cpu`.

### Тайловое декодирование

Большие изображения декодируются частями.

Параметры:

- больший tile быстрее, но требует больше памяти;
- меньший tile снижает использование памяти;
- overlap помогает уменьшить швы;
- `batch_tiles` задаёт количество latent, обрабатываемых за один batch.

### Файлы decoder

Нода ищет decoder в:

```text
ComfyUI/models/vae_approx/
```

Ожидаемые файлы:

```text
taesd_decoder.pth
taesd_decoder.safetensors
taesdxl_decoder.pth
taesdxl_decoder.safetensors
```

Загруженный decoder кешируется.

Формат изображения на выходе:

```text
[B, H, W, C]
```

Для пятимерного latent используется первый temporal/frame dimension.

---

## 9. Fast TAESD Decode Batched V1 Legacy

### Внутреннее имя

```text
LocalFastTAESDDecodeBatchedV1Legacy
```

### Отображаемое имя

```text
Fast TAESD Decode Batched V1 - Legacy
```

Legacy-версия TAESD-ноды, оставленная для совместимости со старыми workflow.

---

## 10. Prompt Randomizer 10 - TXT

### Внутреннее имя

```text
LocalPromptRandomizer10TXT
```

### Отображаемое имя

```text
Prompt Randomizer 10 - TXT
```

### Категория

```text
Local Tools/Prompt
```

Нода выбирает prompt-фрагменты максимум из десяти TXT-файлов.

### Папка со списками

Папка создаётся автоматически:

```text
ComfyUI/custom_nodes/ComfyUI-mini-pack-nodes/prompt_lists/
```

Поддерживаются вложенные папки.

### Формат TXT

Каждая непустая строка считается отдельным вариантом.

Строки, начинающиеся с `#`, считаются комментариями.

Пример:

```text
# Camera angles
close-up portrait
full body
cinematic side view
wide shot
```

### Режимы слотов

Каждый слот поддерживает:

- `disabled`;
- `random`;
- `fixed`.

### Random

Выбирает случайную строку из файла.

### Fixed

Выбирает строку по номеру.

Если номер выходит за пределы файла, он ограничивается доступным диапазоном.

### Воспроизводимость

При включённом `randomize_each_queue`:

- при каждом запуске выбирается новый вариант;
- кеширование ComfyUI обходится.

При отключённом параметре:

- результат воспроизводим;
- выбор зависит от seed.

### Выходы

- итоговый prompt;
- подробный отчёт.

Отчёт содержит:

- seed;
- режим рандомизации;
- имя файла;
- режим слота;
- номер строки;
- выбранный текст.

---

## 11. CLIP Text Encode 2-Part

### Внутреннее имя

```text
LocalCLIPTextEncode2Part
```

### Отображаемое имя

```text
CLIP Text Encode 2-Part
```

### Категория

```text
Local Tools/Conditioning
```

Нода разделяет prompt на смысловые части перед CLIP encoding.

### Входы

- `CLIP`;
- quality prompt;
- character prompt;
- необязательный random prompt.

К random prompt можно подключить:

```text
Prompt Randomizer 10 - TXT
```

### Порядок объединения

1. quality prompt;
2. character prompt;
3. random prompt.

Секции разделяются двумя переводами строки.

### Выходы

- `CONDITIONING`;
- объединённый prompt.

---

# Примеры workflow

## Обычный SDXL workflow

```text
Resolution Preset
    → Empty Latent Image

Prompt Preset
    → CLIP Text Encode

Checkpoint Loader - No VAE
    → MODEL
    → CLIP

Separate VAE Loader
    → VAE

KSampler
    → VAE Decode
```

## Workflow с LoRA

```text
Checkpoint Loader - No VAE
    → LoRA Stack 10
    → KSampler
```

Получение trigger words:

```text
LoRA Trigger Words
    → Prompt Preset
    → CLIP Text Encode
```

## Рандомизация prompt из TXT

```text
Prompt Randomizer 10 - TXT
    → CLIP Text Encode 2-Part
    → KSampler
```

Рекомендуемая структура:

```text
Quality Prompt
Character Prompt
Random TXT Prompt
    → CLIP Text Encode 2-Part
```

## Быстрый preview через TAESD

```text
KSampler
    → Fast TAESD Decode Batched V1
    → Preview Image
```

TAESD лучше использовать для быстрого preview. Для финального результата рекомендуется полный VAE.

---

# Ограничения

- TAESD является approximate decoder и не заменяет полный VAE.
- Для TAESD необходим подходящий decoder в `models/vae_approx`.
- Основная логика сейчас находится в одном большом `__init__.py`.
- Legacy-загрузчик поддерживает необязательный `legacy_nodes.py`.
- В текущий исходник необходимо добавить стандартные импорты, указанные выше.
- Качество trigger words зависит от metadata внутри LoRA.
- Prompt Randomizer работает только с `.txt`.
- Модели и decoder-файлы автоматически не скачиваются.
- Пакет предназначен в первую очередь для image-generation workflow.

---

# Лицензия

GPL-3.0.

---

Сейчас README исходного репозитория содержит только заголовок, поэтому варианты выше являются полноценной документацией на основе фактической реализации `__init__.py`. ([raw.githubusercontent.com](https://raw.githubusercontent.com/UNDEAD2075/ComfyUI-mini-pack-nodes/main/README.md))
