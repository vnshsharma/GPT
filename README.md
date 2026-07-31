# NanoGPT - Building a Bigram Language Model from Scratch

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Learning-green)

---

# 📖 Overview

This repository contains my implementation of the **Bigram Language Model** using **PyTorch**, inspired by **Andrej Karpathy's NanoGPT** project.

The goal of this project is **not simply to train a language model**, but to understand every component that makes modern Large Language Models (LLMs) work.

Rather than using high-level libraries like Hugging Face Transformers, this project starts from the very basics and gradually builds toward GPT.

This repository documents every step of that learning journey.

---

# 📂 Project Structure

```
NanoGPT/
│
├── data/
│   └── input.txt
│
├── model.py
│
└── README.md
```

## Folder Explanation

### data/

Contains the training dataset.

```
data/
    input.txt
```

This text file is the only source of knowledge for the model.

The model learns language patterns solely from this file.

Example

```
Hello World
```

or Shakespeare

```
To be, or not to be...
```

---

### model.py

Contains the complete implementation of

- Character Tokenizer
- Vocabulary Creation
- Batch Generator
- Bigram Language Model
- Loss Function
- Optimizer
- Training Loop
- Text Generation

Everything is written manually without using pretrained models.

---

# Installing

Clone the repository

```bash
git clone https://github.com/yourusername/NanoGPT.git
```

Move into the project

```bash
cd NanoGPT
```

Install PyTorch

```bash
pip install torch
```

---

# Running

Execute

```bash
python model.py
```

---

# Understanding the Code

---

# 1. Import Libraries

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
```

### torch

Provides tensor operations and GPU acceleration.

---

### torch.nn

Contains neural network layers such as

- Linear
- Embedding
- Conv
- LSTM

---

### torch.nn.functional

Contains mathematical operations such as

- Softmax
- Cross Entropy
- ReLU

without creating layers.

---

# 2. Hyperparameters

```python
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
```

These values control the training process.

---

## batch_size

```
32
```

The number of sequences processed simultaneously.

Instead of learning from one sentence at a time,

the model learns from **32 examples together.**

---

## block_size

```
8
```

Maximum context length.

The model can only look at the previous **8 characters** when predicting the next one.

Example

```
Machine
```

If block size is 8

the model only sees

```
Machine
```

before predicting the next character.

---

## max_iters

```
3000
```

Number of optimization steps.

More iterations generally improve learning.

---

## learning_rate

```
0.01
```

Controls how large each optimization step is.

Too large

↓

Training becomes unstable.

Too small

↓

Training becomes very slow.

---

# 3. Device Selection

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

If NVIDIA GPU is available

↓

Training runs on GPU.

Otherwise

↓

CPU is used.

---

# 4. Random Seed

```python
torch.manual_seed(1337)
```

This fixes randomness.

Every run produces the same random numbers.

Useful for

- debugging
- reproducibility
- comparison

---

# 5. Reading Dataset

```python
with open('data/input.txt','r',encoding='utf-8') as f:
    text = f.read()
```

Reads the entire text file into memory.

Suppose

```
Hello
```

Then

```
text = "Hello"
```

---

# 6. Vocabulary Creation

```python
chars = sorted(list(set(text)))
```

Steps

```
Original

banana
```

↓

Unique

```
{a,b,n}
```

↓

Sorted

```
[a,b,n]
```

This becomes our vocabulary.

---

# 7. Character Mapping

```python
stoi
```

String

↓

Integer

Example

```
a → 0

b → 1

n → 2
```

---

```python
itos
```

Integer

↓

Character

```
0 → a

1 → b

2 → n
```

These dictionaries allow conversion in both directions.

---

# 8. Encoding

```python
encode = lambda s:[stoi[c] for c in s]
```

Example

```
cat
```

↓

```
[2,0,18]
```

The neural network only understands numbers.

---

# 9. Decoding

```python
decode = lambda l:''.join(...)
```

Example

```
[2,0,18]
```

↓

```
cat
```

---

# 10. Tensor Conversion

```python
data = torch.tensor(...)
```

Converts the encoded text into a PyTorch tensor.

Neural networks operate on tensors rather than Python lists.

---

# 11. Train Validation Split

```python
90%
```

↓

Training

```
10%
```

↓

Validation

Validation data checks whether the model is learning general language patterns rather than memorizing the training data.

---

# 12. Batch Generation

```python
get_batch()
```

Randomly selects

```
batch_size
```

different positions.

Creates

```
x
```

Current context.

and

```
y
```

Expected next characters.

Example

```
Input

hell

Target

ello
```

---

# 13. Estimate Loss

Every

```
300
```

iterations

the model measures

Training Loss

Validation Loss

to monitor learning.

---

# 14. Bigram Language Model

The heart of the project.

```python
class BigramLanguageModel(nn.Module)
```

---

## Embedding Layer

```python
nn.Embedding(vocab_size,vocab_size)
```

Every character gets its own learnable vector.

Initially

Random.

After training

The vectors learn language statistics.

---

# Forward Pass

Input

```
hello
```

↓

Embedding

↓

Logits

↓

Cross Entropy Loss

↓

Backpropagation

---

# Cross Entropy Loss

```python
loss = F.cross_entropy(...)
```

Measures

How wrong

the predictions are.

Smaller loss

↓

Better predictions.

---

# 15. Text Generation

The model predicts

one character

↓

adds it

↓

predicts again

↓

adds it

↓

continues until

```
max_new_tokens
```

is reached.

---

# 16. Optimizer

```python
AdamW
```

Updates the model parameters using gradients.

This is how learning happens.

---

# 17. Training Loop

Every iteration

```
Get Batch
      ↓
Forward Pass
      ↓
Compute Loss
      ↓
Backward Pass
      ↓
Optimizer Step
      ↓
Repeat
```

---

# Computational Graph

```
input.txt
      │
      ▼
Read Text
      │
      ▼
Vocabulary
      │
      ▼
Encode
      │
      ▼
Tensor
      │
      ▼
Train / Validation Split
      │
      ▼
Random Batch Generator
      │
      ▼
Embedding Layer
      │
      ▼
Logits
      │
      ▼
Cross Entropy Loss
      │
      ▼
Backpropagation
      │
      ▼
AdamW Optimizer
      │
      ▼
Updated Weights
      │
      ▼
Generate Text
```

---

# Learning Outcomes

After completing this project, I understood

- Character-level tokenization
- Vocabulary generation
- Encoding and decoding
- PyTorch tensors
- Embedding layers
- Batch creation
- Forward propagation
- Cross Entropy Loss
- Gradient descent
- Backpropagation
- AdamW optimizer
- Character-level text generation

---

# Future Improvements

This repository currently implements a **Bigram Language Model**.

Future updates will include

- Positional Encoding
- Self Attention
- Multi Head Attention
- Feed Forward Networks
- Residual Connections
- Layer Normalization
- Transformer Blocks
- Complete GPT Architecture
- Improved Text Generation
- Model Checkpointing

---

# Acknowledgements

This implementation is inspired by

**Andrej Karpathy's "Let's build GPT from scratch"**

The purpose of this repository is educational and follows the concepts explained in the NanoGPT project while being implemented independently as part of my own learning journey.

---

# License

This project is intended for educational purposes.