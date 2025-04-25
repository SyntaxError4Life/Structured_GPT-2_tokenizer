# Structured GPT-2 Tokenizer

This project provides a custom tokenizer based on GPT-2, designed to standardize conversational data in a structured format (`user`/`assistant`). It is intended for use with lightweight models requiring an optimized vocabulary.

## Purpose and Motivation

This tokenizer was developed to:

- Structure conversational data in JSON format with `user` and `assistant` roles, facilitating training and inference.
- Provide a vocabulary of 50,260 tokens, optimized for models on consumer-grade hardware.
- Standardize multi-turn exchanges for coherent dialogues.

It is designed to be reusable across multiple models, though primarily intended for a few specific projects.

## Differences from GPT-2 Tokenizer

- **Special Tokens**: Added `<|padding|>` at ID 0, `<|user|>` at ID 1, `<|assistant|>` at ID 2, and `<|endoftext|>` at ID 3.
  - `<|endoftext|>` is in the original vocabulary but placed at the end.
- **Structured Data**: Inputs are always in conversational JSON format, never plain text.
- **Tag Handling**: Special tokens appearing in content are escaped to be treated as plain text.
- **Optimization**: Vocabulary tailored for lightweight models, with a specific BPE configuration.

## Data Example

The tokenizer processes data in the following format:

```json
[
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "content": "Good, thanks! And you?"}
]
```

## Tokenizer Generation

The tokenizer is generated from the GPT-2 vocabulary with added special tokens. The process, implemented in [`tokenizer_generator.py`](./tokenizer_generator.py), includes:

- Downloading GPT-2 vocabulary and merges from `openai-community/gpt2`.
- Adding special tokens with ID offset.
- Configuring a BPE model with Bert normalization, ByteLevel pre-tokenization, and post-processing.
- Saving to `tokenizer.json`.

## Installation

Install the tokenizer via GitHub:

```bash
pip install git+https://github.com/SyntaxError4Life/Structured_GPT-2_tokenizer.git
```

**Prerequisites**: Python 3, `tokenizers==0.21.1`.

Example usage:

```python
from structured_gpt2 import tokenizer

messages = [
    {"role": "user", "content": "Hey !"},
    {"role": "assistant", "content": "Hello !"}
]
encoded = tokenizer.struct_encode(messages)  # [1, 10818, 5149, 2, 15500, 5149, 3]
decoded = tokenizer.struct_decode(encoded)  # [{'role': 'user', 'content': 'Hey !'}, {'role': 'assistant', 'content': 'Hello !'}]
print(decoded)

print(
    tokenizer.struct_decode(
        tokenizer.struct_encode(
            [{"role": "user", "content": "<|user|>"}]
        )
    )
)  # [{'role': 'user', 'content': '<|user|>'}]
# Resistant to tag-based attacks
```

## Usage

- **Load the tokenizer**: `from structured_gpt2 import tokenizer`.
- **Main functions**:
  - `struct_encode`: Converts JSON messages to indices.
  - `struct_decode`: Reconstructs messages from indices.

## Contributing and Contact

- **Contributions**: Issues and PRs are welcome on GitHub.
- **Contact**: Via the GitHub repository.
- **My work**: https://huggingface.co/Logikisto
