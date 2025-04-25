import json
import gc
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.normalizers import BertNormalizer
from tokenizers.processors import TemplateProcessing
from huggingface_hub import hf_hub_download

# Étape 1 : Charger ou recréer le tokenizer
output_file = "tokenizer.json"
try:
    tokenizer = Tokenizer.from_file(output_file)
    print("Tokenizer chargé depuis", output_file)
except Exception as e:
    print(f"Erreur au chargement ({e}). Recréation du tokenizer...")
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
    print(f"Tokenizer recréé et sauvegardé dans {output_file}")

# Étape 2 : Classe pour gérer les messages structurés
class StructuredTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = {
            "user": tokenizer.token_to_id("<|user|>"),
            "assistant": tokenizer.token_to_id("<|assistant|>"),
            "endoftext": tokenizer.token_to_id("<|endoftext|>")
        }
        if None in self.special_ids.values():
            raise ValueError("Un token spécial n’a pas d’ID valide : " + str(self.special_ids))

    def escape_special_tokens(self, text):
        """Échappe les balises spéciales dans le contenu pour les traiter comme texte brut."""
        for token in ["<|user|>", "<|assistant|>", "<|endoftext|>"]:
            escaped_token = token.replace("<", "\\<").replace(">", "\\>")
            text = text.replace(token, escaped_token)
        return text

    def unescape_special_tokens(self, text):
        """Restaure les balises échappées dans le contenu."""
        for token in ["<|user|>", "<|assistant|>", "<|endoftext|>"]:
            escaped_token = token.replace("<", "\\<").replace(">", "\\>")
            text = text.replace(escaped_token, token)
        return text

    def struct_encode(self, messages):
        """Encode une liste de messages en indices."""
        indices = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"].lstrip()
            if role == "user":
                indices.append(self.special_ids["user"])
            elif role == "assistant":
                indices.append(self.special_ids["assistant"])
            else:
                raise ValueError(f"Rôle inconnu : {role}")
            if content:
                # Échapper les balises spéciales pour les traiter comme texte brut
                escaped_content = self.escape_special_tokens(content)
                content_encoded = self.tokenizer.encode(escaped_content).ids
                indices.extend(content_encoded)
        indices.append(self.special_ids["endoftext"])
        return indices

    def struct_decode(self, indices):
        """Décode les indices en messages structurés."""
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

# Étape 3 : Tests exhaustifs
def test_tokenizer():
    struct_tokenizer = StructuredTokenizer(tokenizer)
    
    # Test 1 : Messages avec balises dans le contenu
    messages = [
        {"role": "user", "content": "Salut, j’ai écrit <|user|> dans mon texte"},
        {"role": "assistant", "content": "Pas de souci, c’est bien du texte brut !"}
    ]
    print("\nTest 1 : Balises dans le contenu")
    encoded = struct_tokenizer.struct_encode(messages)
    print("Indices encodés :", encoded)
    decoded = struct_tokenizer.struct_decode(encoded)
    print("Messages décodés :", decoded)
    
    # Test 2 : Messages multiples
    messages = [
        {"role": "user", "content": "Bonjour, comment vas-tu ?"},
        {"role": "assistant", "content": "Je vais bien, merci ! Et toi ?"},
        {"role": "user", "content": "Super, merci !"}
    ]
    print("\nTest 2 : Messages multiples")
    encoded = struct_tokenizer.struct_encode(messages)
    print("Indices encodés :", encoded)
    decoded = struct_tokenizer.struct_decode(encoded)
    print("Messages décodés :", decoded)
    
    # Test 3 : Message unique
    single_message = [{"role": "assistant", "content": "Ceci est un test."}]
    print("\nTest 3 : Message unique")
    encoded = struct_tokenizer.struct_encode(single_message)
    print("Indices encodés :", encoded)
    decoded = struct_tokenizer.struct_decode(encoded)
    print("Messages décodés :", decoded)
    
    # Test 4 : Messages vides
    empty_message = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Réponse vide"}
    ]
    print("\nTest 4 : Messages vides")
    encoded = struct_tokenizer.struct_encode(empty_message)
    print("Indices encodés :", encoded)
    decoded = struct_tokenizer.struct_decode(encoded)
    print("Messages décodés :", decoded)

# Lancer les tests
test_tokenizer()
