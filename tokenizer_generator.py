import json
import gc
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.normalizers import BertNormalizer
from tokenizers.processors import TemplateProcessing
from huggingface_hub import hf_hub_download

# Step 1: Load or recreate the tokenizer
output_file = "tokenizer.json"
try:
    tokenizer = Tokenizer.from_file(output_file)
    print("Tokenizer loaded from", output_file)
except Exception as e:
    print(f"Error loading tokenizer ({e}). Recreating tokenizer...")
    repo_id = "openai-community/gpt2"
    vocab_file = hf_hub_download(repo_id=repo_id, filename="vocab.json")
    merges_file = hf_hub_download(repo_id=repo_id, filename="merges.txt")
    
    with open(vocab_file, "r", encoding="utf-8") as f:
        original_vocab = json.load(f)
    
    special_tokens = {
        "<|padding|>": 0,
        "<|user|>": 1,
        "<|assistant|>": 2,
        "<|endoftext|>": 3
    }
    
    new_vocab = special_tokens.copy()
    offset = len(special_tokens)
    for token, id in original_vocab.items():
        if token not in special_tokens:
            new_vocab[token] = id + offset if id < 50256 else id + offset - 1
    
    with open(merges_file, "r", encoding="utf-8") as f:
        merges = [tuple(line.strip().split()) for line in f if line.strip() and not line.startswith("#")]
    
    bpe_model = models.BPE(vocab=new_vocab, merges=merges)
    tokenizer = Tokenizer(bpe_model)
    tokenizer.add_special_tokens(list(special_tokens.keys()))
    tokenizer.normalizer = BertNormalizer(lowercase=False)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = TemplateProcessing(
        single="$0",
        pair="$A $B",
        special_tokens=[(token, new_vocab[token]) for token in special_tokens]
    )
    tokenizer.save(output_file)
    print(f"Tokenizer recreated and saved to {output_file}")

# Step 2: Class to handle structured messages
class StructuredTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = {
            "user": tokenizer.token_to_id("<|user|>"),
            "assistant": tokenizer.token_to_id("<|assistant|>"),
            "endoftext": tokenizer.token_to_id("<|endoftext|>")
        }
        if None in self.special_ids.values():
            raise ValueError("A special token has an invalid ID: " + str(self.special_ids))

    def escape_special_tokens(self, text):
        """Escape special tokens in the content to treat them as plain text."""
        for token in ["<|user|>", "<|assistant|>", "<|endoftext|>"]:
            escaped_token = token.replace("<", "\\<").replace(">", "\\>")
            text = text.replace(token, escaped_token)
        return text

    def unescape_special_tokens(self, text):
        """Restore escaped tags in the content."""
        for token in ["<|user|>", "<|assistant|>", "<|endoftext|>"]:
            escaped_token = token.replace("<", "\\<").replace(">", "\\>")
            text = text.replace(escaped_token, token)
        return text

    def struct_encode(self, messages):
        """Encode a list of messages into indices."""
        indices = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"].lstrip()
            if role == "user":
                indices.append(self.special_ids["user"])
            elif role == "assistant":
                indices.append(self.special_ids["assistant"])
            else:
                raise ValueError(f"Unknown role: {role}")
            if content:
                escaped_content = self.escape_special_tokens(content)
                content_encoded = self.tokenizer.encode(escaped_content).ids
                indices.extend(content_encoded)
        indices.append(self.special_ids["endoftext"])
        return indices

    def struct_decode(self, indices):
        """Decode indices into structured messages."""
        messages = []
        current_role = None
        current_content_ids = []
        for idx in indices:
            if idx == self.special_ids["user"]:
                if current_role:
                    content = self.unescape_special_tokens(self.tokenizer.decode(current_content_ids)).lstrip()
                    messages.append({"role": current_role, "content": content})
                current_role = "user"
                current_content_ids = []
            elif idx == self.special_ids["assistant"]:
                if current_role:
                    content = self.unescape_special_tokens(self.tokenizer.decode(current_content_ids)).lstrip()
                    messages.append({"role": current_role, "content": content})
                current_role = "assistant"
                current_content_ids = []
            elif idx == self.special_ids["endoftext"]:
                if current_role:
                    content = self.unescape_special_tokens(self.tokenizer.decode(current_content_ids)).lstrip()
                    messages.append({"role": current_role, "content": content})
                break
            else:
                current_content_ids.append(idx)
        return messages

# Step 3: Comprehensive tests
def test_tokenizer():
    struct_tokenizer = StructuredTokenizer(tokenizer)
    
    # Test 1: Messages with tags in content
    messages = [
        {"role": "user", "content": "Hi, I wrote <|user|> in my text"},
        {"role": "assistant", "content": "No issue, it’s plain text!"}
    ]
    print("\nTest 1: Tags in content")
    encoded = struct_tokenizer.struct_encode(messages)
    print("Encoded indices:", encoded)
    decoded = struct_tokenizer.struct_decode(encoded)
    print("Decoded messages:", decoded)
    
    # Test 2: Multiple messages
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm good, thanks! And you?"},
        {"role": "user", "content": "Great, thanks!"}
    ]
    print("\nTest 2: Multiple messages")
    encoded = struct_tokenizer.struct_encode(messages)
    print("Encoded indices:", encoded)
    decoded = struct_tokenizer.struct_decode(encoded)
    print("Decoded messages:", decoded)
    
    # Test 3: Single message
    single_message = [{"role": "assistant", "content": "This is a test."}]
    print("\nTest 3: Single message")
    encoded = struct_tokenizer.struct_encode(single_message)
    print("Encoded indices:", encoded)
    decoded = struct_tokenizer.struct_decode(encoded)
    print("Decoded messages:", decoded)
    
    # Test 4: Empty messages
    empty_message = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Empty response"}
    ]
    print("\nTest 4: Empty messages")
    encoded = struct_tokenizer.struct_encode(empty_message)
    print("Encoded indices:", encoded)
    decoded = struct_tokenizer.struct_decode(encoded)
    print("Decoded messages:", decoded)

# Run tests
test_tokenizer()
