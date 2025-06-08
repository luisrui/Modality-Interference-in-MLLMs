# Diagnosing and Mitigating Modality Interference in Multimodal Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2505.19616-b31b1b.svg)](https://arxiv.org/abs/2505.19616)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This repository contains the official implementation of our paper:

> **Diagnosing and Mitigating Modality Interference in Multimodal Large Language Models**  
> Rui Cai, Bangzheng Li, Xiaofei Wen, Muhao Chen, Zhe Zhao  
> [arXiv:2505.19616](https://arxiv.org/abs/2505.19616)

---

## 🔍 Overview

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across tasks, yet they often exhibit difficulty in distinguishing task-relevant from irrelevant signals—particularly in tasks like Visual Question Answering (VQA)—which can lead to susceptibility to misleading or spurious inputs. We refer to this broader limitation as the Cross-Modality Competency Problem—the model’s inability to fairly evaluate all modalities. This vulnerability becomes more evident in modality-specific tasks—such as image classification or pure text question answering—where models are expected to rely solely on one modality. In such tasks, spurious information from irrelevant modalities often lead to significant performance degradation. We refer to this failure as Modality Interference, which serves as a concrete and measurable instance of the cross-modality competency problem, and we further design a perturbation-based causal diagnostic experiment to verify and quantify this problem. To mitigate modality interference, we propose a novel framework to finetune MLLMs, including perturbation-based data augmentations with both heuristic perturbations and adversarial perturbations via Projected Gradient Descent (PGD), and a consistency regularization strategy applying on model outputs with original and perturbed inputs. Experiments on multiple benchmark datasets (image-heavy, text-heavy and VQA tasks) and multiple model families with different scales demonstrate significant improvements in robustness and cross-modality competency, indicating our method’s effectiveness in boosting unimodal reasoning ability while enhancing performance on multimodal tasks.

For more details, please refer to our [paper](https://arxiv.org/abs/2505.19616).

---

## 🛠 Installation

We provide separate environments for different model families:

<details>
<summary><strong>For LLaVA-1.5 and InstructBLIP-vicuna</strong></summary>

```bash
conda create -n mllm_llava -f MLLM.yml
conda activate mllm_llava
```
</details>

<details>
<summary><strong>For Qwen2.5-VL</strong></summary>

```bash
conda create -n mllm_qwen -f vllm.yml
conda activate mllm_qwen
```
</details>


## 📦 Dataset

All training and evaluation data used in our experiments is publicly available at:

👉 HuggingFace: [luisrui/training_data](https://huggingface.co/datasets/luisrui/training_data)

## 🚀 Training

To train a model with different settings (example: LLaVA-1.5-13B), use the following command:

```bash
deepspeed src/llava/llava_consistent.py \
  --config configs/model_train/llava-v1.5-7b/args_full_KL_PGD.yaml
```

You may switch configs to enable:
	•	PGD adversarial training
	•	KL or JS consistency regularization
	•	Image-heavy, Text-heavy, or Mixed dataset sampling ratio settings

More examples and configs are available in configs/model_train/.

## 📊 Evaluation

Evaluation scripts are provided in analysis/ and support:
	•	Unimodal perturbation analysis
	•	VQA and classification accuracy

You can directly run follwing command for prediction on multiple datatsets:
```python
bash zs_inference.sh 
    --model_name llava-1.5-7b
    --checkpoint_path path_to_your_checkpoint
    --batch_size batch_size
    --tag your_experiment_setting
    --all
```
For evaluation, you can run
```python
bash evaluate.sh 
    --model_name llava-1.5-7b
    --tag same_tag_as_your_experiment_setting
    --all
```
We also provide pretrained checkpoints and a unified evaluation interface at
👉 HuggingFace: [luisrui/](https://huggingface.co/luisrui)

## 📄 Citation

If you find this repository helpful in your research, please cite our paper:\

```bibtex
@article{cai2025diagnosing,
  title={Diagnosing and Mitigating Modality Interference in Multimodal Large Language Models},
  author={Cai, Rui and Li, Bangzheng and Wen, Xiaofei and Chen, Muhao and Zhao, Zhe},
  journal={arXiv preprint arXiv:2505.19616},
  year={2025}
}
```