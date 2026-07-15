# NanoGPT: Building a GPT from Scratch in PyTorch

## Overview

This repository documents the step-by-step implementation of a character-level language model using PyTorch, following the educational concepts presented by Andrej Karpathy in his NanoGPT series.

The objective of this project is not merely to reproduce an existing implementation, but to gain a complete understanding of every component involved in constructing a modern Generative Pre-trained Transformer (GPT) architecture from first principles.

The project begins with the simplest possible language model—the Bigram Language Model—and will gradually evolve into a complete Transformer-based GPT capable of generating coherent text.

Every stage of development is implemented manually with minimal abstractions to emphasize understanding over convenience.

---

# Project Objectives

The primary goals of this project are:

- Understand how language models represent text.
- Learn how tokenization works at the character level.
- Build datasets suitable for neural language modeling.
- Understand embeddings and why they are useful.
- Learn how autoregressive text generation works.
- Implement loss computation using Cross Entropy.
- Develop intuition behind Transformer architectures.
- Progress toward implementing a complete GPT model from scratch.

---

# Current Implementation

The repository currently includes the complete implementation of a Bigram Language Model.

Implemented components include:

- Reading raw text data
- Vocabulary construction
- Character-level tokenization
- Encoding and decoding utilities
- Tensor conversion
- Dataset splitting
- Batch generation
- Context-target preparation
- Bigram Language Model
- Cross Entropy loss computation
- Autoregressive text generation

---

# Repository Structure

```
.
├── input.txt
├── model.py
└── README.md
```

| File | Description |
|------|-------------|
| input.txt | Training corpus |
| model.py | Complete implementation of the Bigram Language Model |
| README.md | Project documentation |

---

# Character-Level Tokenization

Unlike word-level tokenizers, this project begins with a character-level vocabulary.

Each unique character appearing in the dataset is assigned a unique integer identifier.

Example:

```
Vocabulary

abcdefghijklmnopqrstuvwxyz .,!?'
```

Character-to-index mapping:

```python
'a' -> 0
'b' -> 1
'c' -> 2
...
```

Index-to-character mapping:

```python
0 -> 'a'
1 -> 'b'
2 -> 'c'
...
```

Two helper functions are created:

```python
encode(text)
```

Converts text into integer tokens.

Example:

```
hello

↓

[7,4,11,11,14]
```

```python
decode(tokens)
```

Converts integer tokens back into readable text.

```
[7,4,11,11,14]

↓

hello
```

---

# Dataset Preparation

The complete corpus is converted into a one-dimensional PyTorch tensor.

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

Example:

```
Input Text

hello world

↓

Tensor

[7,4,11,11,14,26,22,14,17,11,3]
```

This tensor becomes the foundation for all subsequent training.

---

# Training and Validation Split

To evaluate the model objectively, the dataset is divided into two independent subsets.

Training Set

- Used for learning model parameters.

Validation Set

- Used only to evaluate performance.

Current split:

```
90% Training
10% Validation
```

Implementation:

```python
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]
```

---

# Batch Generation

Rather than processing the entire dataset simultaneously, training is performed using randomly sampled mini-batches.

Each batch consists of multiple sequences of fixed length.

Example

Context

```
hello wo
```

Target

```
ello wor
```

Every character attempts to predict the next character.

For example

| Input | Target |
|--------|--------|
| h | e |
| e | l |
| l | l |
| l | o |
| o | (space) |

This transforms language modeling into a supervised learning problem.

---

# Bigram Language Model

The Bigram Language Model is the simplest possible neural language model.

Its prediction depends only on the current token.

Mathematically,

```
P(next token | current token)
```

No previous context is considered.

The model contains a single learnable embedding matrix.

```python
nn.Embedding(vocab_size, vocab_size)
```

Each row corresponds to one vocabulary token.

Each row stores the logits representing the probability distribution of the next possible character.

Conceptually,

```
Current Character

↓

Embedding Lookup

↓

Vocabulary Logits

↓

Softmax

↓

Probability Distribution

↓

Sample Next Character
```

No attention mechanism is used.

No positional information is used.

No hidden layers are present.

The model learns only transition probabilities between adjacent characters.

---

# Forward Pass

During training, the model receives

```
Input Tokens

↓

Embedding Lookup

↓

Logits

↓

Cross Entropy Loss
```

The logits have dimensions

```
(Batch Size,
 Context Length,
 Vocabulary Size)
```

Before computing the loss, the tensors are reshaped into

```
(B × T, Vocabulary Size)
```

and

```
(B × T)
```

to satisfy the requirements of PyTorch's Cross Entropy Loss.

---

# Loss Function

The training objective is to minimize Cross Entropy Loss.

```python
loss = F.cross_entropy(logits, targets)
```

Cross Entropy measures how different the predicted probability distribution is from the true target distribution.

Lower loss indicates better predictions.

---

# Text Generation

Text generation is autoregressive.

The process is repeated until the desired sequence length is reached.

Algorithm

1. Feed the current sequence into the model.
2. Compute logits.
3. Select the logits corresponding to the final time step.
4. Apply Softmax to obtain probabilities.
5. Sample the next token.
6. Append the sampled token.
7. Repeat.

This process enables the model to generate arbitrary-length sequences after training.

---

# Current Limitations

The current model is intentionally simple.

Its limitations include:

- No contextual understanding.
- No positional information.
- No self-attention.
- No multi-head attention.
- Cannot model long-range dependencies.
- Limited expressive power.
- Learns only immediate character transitions.

Despite these limitations, it establishes the mathematical and implementation foundation required for Transformer models.

---

# Training the Bigram Language Model

After building the model, we train it so it can learn to predict the next character by minimizing the prediction error (loss).

---

## Create the Optimizer

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
```

- **AdamW** updates the model's weights after every training step.
- `m.parameters()` returns all learnable parameters of the model.
- `lr=1e-3` is the learning rate, which controls how much the weights change during each update.

---

## Training Loop

```python
batch_size = 32

for steps in range(100):
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

### Training Steps

1. **Get a Batch**
   ```python
   xb, yb = get_batch("train")
   ```
   Randomly samples training data.

2. **Forward Pass**
   ```python
   logits, loss = m(xb, yb)
   ```
   Computes predictions and the Cross Entropy Loss.

3. **Clear Old Gradients**
   ```python
   optimizer.zero_grad(set_to_none=True)
   ```
   Removes gradients from the previous iteration.

4. **Backpropagation**
   ```python
   loss.backward()
   ```
   Computes gradients for all model parameters.

5. **Update Weights**
   ```python
   optimizer.step()
   ```
   Updates the model using the computed gradients.

---

## Print Final Loss

```python
print(loss.item())
```

Displays the final training loss. Lower loss generally indicates better learning.

---

## Generate Text

```python
print(
    decode(
        m.generate(
            idx=torch.zeros((1,1), dtype=torch.long),
            max_new_tokens=500
        )[0].tolist()
    )
)
```

After training, the model generates **500 new characters**, which are decoded back into readable text.

---

## Training Pipeline

```
Training Data
      │
      ▼
Random Batch
      │
      ▼
Forward Pass
      │
      ▼
Compute Loss
      │
      ▼
Backpropagation
      │
      ▼
Update Weights
      │
      ▼
Repeat
      │
      ▼
Generate Text
```

# Future Development

The project will gradually evolve into a complete GPT implementation.

Planned milestones include

- Token Embeddings
- Positional Embeddings
- Self-Attention
- Scaled Dot Product Attention
- Multi-Head Attention
- Feed Forward Networks
- Residual Connections
- Layer Normalization
- Dropout
- Transformer Blocks
- Training Loop
- Evaluation Pipeline
- Checkpoint Saving
- Model Sampling
- Complete GPT Architecture

---

# Project Status

Current Stage

```
Bigram Language Model
```

The repository will continue to be updated as additional components of the Transformer architecture are implemented.