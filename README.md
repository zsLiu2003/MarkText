# MarkText: Comprehensive Text Watermarking Framework

MarkText is a comprehensive framework for text watermarking research, providing implementations of multiple state-of-the-art watermarking algorithms and evaluation tools for language model-generated text.

## 🚀 Features

- **Multiple Watermarking Algorithms**: Supports 14+ different watermarking schemes including KGW, Exponential, Unigram, SIR, SWEET, UPV, and more
- **Attack Evaluation**: Built-in robustness testing against various text attacks
- **Quality Assessment**: Comprehensive text quality evaluation metrics
- **Efficiency Analysis**: Performance benchmarking and efficiency measurement tools
- **Visualization**: Rich visualization tools for watermark analysis and results presentation

## 📁 Project Structure

```
marktext/
├── core/                   # Core functionality
│   ├── generate.py        # Text generation with watermarks
│   ├── detect.py          # Watermark detection
│   ├── main.py            # Main entry point
│   ├── quality.py         # Quality evaluation
│   ├── efficiency.py      # Performance benchmarking
│   └── utils/             # Utility functions
├── watermarks/            # Watermarking algorithms
│   ├── kgw/              # KGW watermarking
│   ├── exponential/      # Exponential watermarking
│   ├── unigram/          # Unigram watermarking
│   ├── sir/              # SIR watermarking
│   ├── sweet/            # SWEET watermarking
│   ├── upv/              # UPV watermarking
│   └── ...               # Additional schemes
├── attacks/              # Attack implementations
│   └── attacks/          # Various text attack methods
├── evaluation/           # Evaluation tools and pipelines
│   ├── pipelines/        # Evaluation pipelines
│   └── tools/            # Analysis tools
├── datasets/             # Test datasets
│   ├── c4/               # C4 dataset samples
│   ├── human_eval/       # HumanEval dataset
│   └── wmt16_de_en/      # WMT16 translation dataset
├── configs/              # Configuration files
├── tests/                # Test scripts
├── scripts/              # Utility scripts
└── visualize/            # Visualization tools
```

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- Conda or Mamba package manager

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MarkText
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f marktext/environment.yml
   conda activate ADS
   ```

3. **Install additional dependencies**:
   ```bash
   pip install transformers torch torchvision torchaudio
   ```

## 🚀 Quick Start

### Basic Watermark Generation

```python
from marktext.core.generate import generate_model_base
from marktext.core.util.classes import ConfigSpec

# Configure watermarking
config = ConfigSpec(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    watermark_name="KGW"
)

# Generate watermarked text
watermark_list = ["sample text to watermark"]
result = generate_model_base(config, watermark_list)
```

### Watermark Detection

```python
from marktext.core.detect import detect

# Detect watermarks in text
result = detect(config, "path/to/generated/text.json", "no_attack")
print(f"Watermark detection rate: {result['watermark_percent']}")
```

### Command Line Usage

```bash
# Generate watermarked text
python marktext/core/main.py marktext/configs/KGW.json

# Run evaluation pipeline
python marktext/tests/test_pipeline.py
```

## 📊 Supported Watermarking Algorithms

### Pre-text Watermarks
- **[KGW](https://arxiv.org/abs/2301.10226)**: Kirchenbauer-Geiping-Wen watermarking
- **[Unigram](https://arxiv.org/abs/2306.17439)**: Unigram frequency manipulation
- **[Convert](https://eprint.iacr.org/2023/763)**: Conversion-based watermarking
- **[Inverse](https://arxiv.org/abs/2307.15593)**: Inverse sampling technique
- **[Exponential](https://www.scottaaronson.com/talks/watermark.ppt)**: Exponential distribution-based

### Post-text Watermarks
- **[WHITEMARK](https://arxiv.org/abs/2310.08920)**: Format-based watermarking
- **[UniSpaCh](https://www.researchgate.net/publication/256991692_UniSpaCh_A_text-based_data_hiding_method_using_Unicode_space_characters)**: Unicode space character manipulation
- **[Linguistic](https://arxiv.org/abs/2305.08883)**: Linguistic feature-based watermarking

### Additional Algorithms
- **SIR**: Syntax-aware watermarking
- **SWEET**: Semantic-preserving technique
- **UPV**: Universal paraphrase-resistant
- **EXP**: Enhanced exponential watermarking
- **EWD**: Extended watermark detection

## 🔧 Configuration

### Algorithm Configuration

Each watermarking algorithm can be configured through JSON files in `marktext/configs/`:

```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "watermark_name": "KGW",
  "watermark_config": {
    "gamma": 0.25,
    "delta": 2.0,
    "seeding_scheme": "simple_1"
  }
}
```

## 🛡️ Attack Methods

The framework includes **12 attack methods** to test watermark robustness:

### Pre-text Attacks
- **[Emoji attack](https://x.com/goodside/status/1610682909647671306)**: Emoji injection
- **[Distill](https://arxiv.org/pdf/2312.04469)**: Knowledge distillation attack

### Post-text Attacks
- **[Contraction](https://arxiv.org/abs/2211.09110)**: Text contraction
- **[Expansion](https://arxiv.org/abs/2211.09110)**: Text expansion
- **[Lowercase](https://arxiv.org/abs/2211.09110)**: Case modification
- **[Misspelling](https://arxiv.org/abs/2211.09110)**: Intentional misspellings
- **[Typo](https://arxiv.org/abs/2211.09110)**: Typographical errors
- **[Modify](https://arxiv.org/abs/2312.00273)**: Text modification
- **[Synonym](https://arxiv.org/abs/2312.00273)**: Synonym replacement
- **[Paraphrase](https://arxiv.org/abs/2303.13408)**: Paraphrasing attack
- **[Translation](https://github.com/argosopentech/argos-translate)**: Back-translation
- **[Token attack](https://arxiv.org/abs/2307.15593)**: Token-level manipulation

### Usage Examples

```bash
# Single attack
python marktext/attacks/attack.py your_config_path

# Multi-attack
python marktext/attacks/attack_multi.py your_config_path

# Emoji attack
python marktext/attacks/attack_emoji.py your_config_path
```

## 📈 Evaluation & Quality Assessment

### Quality Evaluation

```bash
# Evaluate text quality
python marktext/core/quality.py your_config_path

# Calculate quality metrics
python marktext/evaluation/tools/quality_calcu.py your_config_path
```

### Performance Benchmarking

```bash
# Benchmark efficiency
python marktext/core/efficiency.py your_config_path
```

### Detection Evaluation

```bash
# Detect watermarks
python marktext/core/detect.py your_config_path
```

## 🔬 Research & Citation

This framework supports research in text watermarking. If you use MarkText in your research, please cite:

```bibtex
@article{liu2024evaluating,
  title={On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks},
  author={Liu, Zesen and Cong, Tianshuo and He, Xinlei and Li, Qi},
  journal={arXiv preprint arXiv:2407.04794},
  year={2024}
}
```