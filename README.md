# NAACL-2024-PiLM

This repo contains code corresponding to the paper `Plug-in Language Model: Controlling Text Generation with a Simple Regression Model` https://aclanthology.org/2024.findings-naacl.139/, published at NAACL 2024 Findings.



### Environment Setup

```bash
pip install -r requirements.txt
```

#### Quick Start

```bash
cd {task}
bash run_PiLM-RL.sh
```

- task: `sentiment_control`, `topic_control`, `language_detoxification`
- generate text by `PiLM-RL` and save unmodify/modify for training Controller

```bash
bash run_train_Controller.sh
```

- train Controller

```bash
bash run_PiLM-Controller.sh
```

- generate text by `PiLM-Controller`



---

pre-calculate embedding of wordlist

```bash
python preprocess_wordlists_embedding.py \
--file "./data/word_list/topic_words.json" \
--output_path "./data/word_list/topic_words_embeddings.pickle" \
```



## Citation

```
@inproceedings{yang-etal-2024-plug,
    title = "Plug-in Language Model: Controlling Text Generation with a Simple Regression Model",
    author = "Yang, Nai-Chi  and
      Ma, Wei-Yun  and
      Cheng, Pu-Jen",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.139",
    pages = "2165--2181",
    abstract = "Large-scale pre-trained language models have displayed unrivaled capacity in generating text that closely resembles human-written text. Nevertheless, generating texts adhering to specific conditions without fine-tuning or adding new parameters can be challenging. Contemporary approaches commonly rely on either prompts or auxiliary models to avoid modifying the language models. These auxiliary models are designed to assess whether a generated token contributes to meeting the desired requirements. These approaches adjust the distribution of the next token during the inference phase by leveraging the prediction score of the desired attribute to calculate gradients. However, these auxiliary models typically require the language model{'}s latent states. This prerequisite challenges integrating various existing black box attribute models or tools. We present the Plug-in Language Model (PiLM) as a solution to address the limitations. PiLM leverages reinforcement learning to utilize black box tools directly, adjusting the latent state to control text generation. However, performing backpropagation during the inference phase is time-consuming for PiLM. By replacing backpropagation with a simple regression model, PiLM can achieve an inference time comparable to that of the original LLM. Experiment results show that our approaches in this paper outperform existing state-of-the-art methods that rely on gradient-based, weighted decoding, or prompt-based methodologies.",
}
```

