# CF-VLM: CounterFactual Vision-Language Fine-tuning

This repository contains research code and example data for:

**CF-VLM: CounterFactual Vision-Language Fine-tuning**  
Jusheng Zhang, Kaitong Cai, Yijia Fan, Jian Wang, Keze Wang  
NeurIPS 2025

Paper: [NeurIPS 2025 Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/file/541b6d155146d142a5e10d787d1d430f-Paper-Conference.pdf)

## Overview

CF-VLM improves the causal reasoning ability of vision-language models by fine-tuning with counterfactual samples. The code includes scripts for generating counterfactual data, fine-tuning Qwen2.5-VL with LoRA, and evaluating CLIP-based similarity.

## Files

- `finetune.py`: CF-VLM fine-tuning with counterfactual objectives.
- `train.py`: baseline Qwen2.5-VL LoRA fine-tuning.
- `process.py`: counterfactual text generation utilities.
- `QWEN2.5.py`: Qwen2.5-VL based counterfactual data generation and fine-tuning script.
- `clip_best.py`: CLIP-based scoring/selection utility.
- `data/`: small example counterfactual samples.

## Installation

```bash
pip install -r requirements.txt
```

The scripts expect a local Qwen2.5-VL checkpoint. Update `model_path` in the scripts to your local model path before running.

## Citation

```bibtex
@inproceedings{zhang2025cfvlm,
  title={CF-VLM: CounterFactual Vision-Language Fine-tuning},
  author={Zhang, Jusheng and Cai, Kaitong and Fan, Yijia and Wang, Jian and Wang, Keze},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
