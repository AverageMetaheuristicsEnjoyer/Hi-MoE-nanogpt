# Hi-MoE for nanoGPT on OpenWebText

## install

```sh
pip install torch numpy transformers datasets==3.6.0 tiktoken wandb tqdm matplotlib seaborn
git clone https://github.com/microsoft/Tutel.git third_party/tutel
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3
-  `tutel` for expert parallelism (the repo falls back to `third_party/tutel` if a system install is unavailable)

#### Prepare OpenWebText

We first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText used to train GPT-2:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes.

## acknowledgements

Thank you to Andrej Karpathy to providing an awesome [starting point](https://github.com/karpathy/nanoGPT) for the nanoMoE implementation!

Thank you to the author of this [blog](https://cameronrwolfe.substack.com/p/nano-moe) upon our code is based.
