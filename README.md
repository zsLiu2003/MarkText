## On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks

This is the official implementation of our paper **[On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks](11)**

<img src="C:\Users\20191\AppData\Roaming\Typora\typora-user-images\image-20240630150448703.png" alt="image-20240630150448703" style="zoom: 80%;" />

## Supported methods

#### A. Pre-text watermark

- [KGW](https://arxiv.org/abs/2301.10226)
- [Unigram](https://arxiv.org/abs/2306.17439)
- [Convert](https://eprint.iacr.org/2023/763)
- [Inverse](https://arxiv.org/abs/2307.15593)
- [Exponential](https://www.scottaaronson.com/talks/watermark.ppt)

#### B. Post-text watermark

- [WHITEMARK](https://arxiv.org/abs/2310.08920)
- [UniSpaCh](https://www.researchgate.net/publication/256991692_UniSpaCh_A_text-based_data_hiding_method_using_Unicode_space_characters)
- [Linguistic ](https://arxiv.org/abs/2305.08883)

### Preparation

```bash
git clone https://github.com/zsLiu2003/MarkText.git;
cd MarkText;
conda env create -f environment.yml;
conda activate Mi;
```

### Watermarking

The code of Linguistic  is from [Text_watermark](https://github.com/Kiode/Text_Watermark), the watermark code of KGW, Unigram, Inverse, Exponential are from [MarkLLM](https://github.com/THU-BPM/MarkLLM), and our code of Convert watermark is from [MarkMyWords](https://github.com/wagner-group/MarkMyWords).

- Generate Watermarked text for  [WHITEMARK](https://arxiv.org/abs/2310.08920), [UniSpaCh](https://www.researchgate.net/publication/256991692_UniSpaCh_A_text-based_data_hiding_method_using_Unicode_space_characters), [Linguistic ](https://arxiv.org/abs/2305.08883).

  ```bash
  python main.py your_config_path #It is the configuretion setting to the watermark.
  ```

- Generate watermarked text for [KGW](https://arxiv.org/abs/2301.10226), [Unigram](https://arxiv.org/abs/2306.17439), [Convert](https://eprint.iacr.org/2023/763), [Inverse](https://arxiv.org/abs/2307.15593), [Exponential](https://www.scottaaronson.com/talks/watermark.ppt).

  ```bash
  #For KGW, Unigram, Inverse and Exponential
  python generate.py your_config_path
  
  #For Convert
  cd MarkMyWords;
  bash install.sh;
  watermark-benchmark-run config.yml #Please set the watermark name to Binary which is the Convert watermark scheme we used.
  ```

- The format of all watermarked text.

  ```python
  @dataclass(frozen=False)
  class Generation:
      watermark: Optional[WatermarkConfig] = None #watermark config
      watermark_name: str = '' #watermark name
      key: Optional[int] = None #watermark key which is a Pseudo-random number
      attack: Optional[str] = None #attack_name
      id: int = 0 #text id
      prompt: str = '' #the model input
      response: str = '' #the model output(watermarked/unwatermarked)
      quality: Optional[float] = None #the quality of the text
      token_count: int = 0 #the number of tokens that we need to verify the watermark
      temperature: float = 1.0 #the training temperature
  ```

- The format of watermark config.

  ```python
  class ConfigSpec:
      model_name: str = '' #the model name
      watermark_name: str = '' #watermark name
      watermark_path: str = '' #watermark path
      output_path: str = '' #the output path of watermarked text or attacked text
      prompt_path: str = '' #the input path of prompt dataset
      #trainin setting 
      batch_size: int = 8
      temperature: float = 1.0
      devices: Optional[List[int]] = None
      misspellings: str = "MarkMyWords/run/static_data/misspellings.json" #the dataset of mispelling attack 
  ```

### Attacking

We have implemented **twelve** attack methods to remove watermark.

#### A. Pre-text attacks

- [Emoji attack](https://x.com/goodside/status/1610682909647671306)
- [Distill](https://arxiv.org/pdf/2312.04469)

#### B. Post-text attacks

- [Contraction](https://arxiv.org/abs/2211.09110)
- [Expansion](https://arxiv.org/abs/2211.09110)
- [Lowercase](https://arxiv.org/abs/2211.09110)
- [Misspelling](https://arxiv.org/abs/2211.09110)
- [Typo](https://arxiv.org/abs/2211.09110)
- [Modify](https://arxiv.org/abs/2312.00273)
- [Synonym](https://arxiv.org/abs/2312.00273)
- [Paraphrase](https://arxiv.org/abs/2303.13408)
- [Translation](https://github.com/argosopentech/argos-translate)
- [Token attack](https://arxiv.org/abs/2307.15593)

#### C. Attack all watermark texts

- For emoji attack. 

  ```bash
  python attack_emoji.py your_config_path #Your should set your watermark list
  ```

- For distill attack.

  ```bash
  cd watermark_distill;
  
  # For Logit-based watermark distillation(e.g. KGW, Unigram)
  bash scripts/train/train_llama_logit_distill.sh <watermark_type> <output_dir/> <master_port> <llama_path>
  # For Sampling modification-based watermark distillation(e.g. Inverse, Convert, Exponential)
  bash scripts/train/generate_sampling_distill_train_data.sh <watermark_type> <llama_path>
  
  # For evaluation
  bash scripts/evaluate/generate_and_evaluate.sh <dataset> <output_file> <llama_path> <perplexity_model> [models]
  ```

- For the other all attacks.

  ```bash
  python attack.py your_config_path
  ```

#### D. Multi-attack

```bash
python attack_multi.py your_config_path
```

### Detection

A. Detect watermark of Format, UniSpaCh, Lexical

```bash
python detect.py your_config_path
```

B. Detect watermark of KGW, Unigram, Inverse, Exponential

```bash
# Please add the detect function:
is_watermark,score_result = myWatermark.detect_watermark(watermarked_text)
# And execute the following command
python generate.py your_config_path
```

### Quality

```bash
1. python quality.py your_config_path
2. python quality_calcu.py your_config_path
```

