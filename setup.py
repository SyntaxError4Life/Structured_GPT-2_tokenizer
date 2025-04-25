from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import shutil

# Définition des chemins
package_name = "structured_gpt2"
package_dir = package_name
init_file = os.path.join(package_dir, "__init__.py")
tokenizer_file = "tokenizer.json"
dest_tokenizer_file = os.path.join(package_dir, "tokenizer.json")

# Création du répertoire du package
os.makedirs(package_dir, exist_ok=True)

# Contenu de __init__.py
init_content = """import os
from tokenizers import Tokenizer

# Chemin vers tokenizer.json dans le package
tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")

# Chargement du tokenizer
try:
    tokenizer = Tokenizer.from_file(tokenizer_path)
except Exception as e:
    raise RuntimeError(f"Erreur au chargement du tokenizer : {e}")

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
        \"\"\"Échappe les balises spéciales pour les traiter comme texte brut.\"\"\"
        for token in ["<|user|>", "<|assistant|>", "<|endoftext|>"]:
            escaped_token = token.replace("<", "\\\\<").replace(">", "\\\\>")
            text = text.replace(token, escaped_token)
        return text

    def unescape_special_tokens(self, text):
        \"\"\"Restaure les balises échappées.\"\"\"
        for token in ["<|user|>", "<|assistant|>", "<|endoftext|>"]:
            escaped_token = token.replace("<", "\\\\<").replace(">", "\\\\>")
            text = text.replace(escaped_token, token)
        return text

    def struct_encode(self, messages):
        \"\"\"Encode une liste de messages en indices.\"\"\"
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
                escaped_content = self.escape_special_tokens(content)
                content_encoded = self.tokenizer.encode(escaped_content).ids
                indices.extend(content_encoded)
        indices.append(self.special_ids["endoftext"])
        return indices

    def struct_decode(self, indices):
        \"\"\"Décode les indices en messages structurés.\"\"\"
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

# Exposition de l'objet tokenizer
tokenizer = StructuredTokenizer(tokenizer)
"""

# Écriture de __init__.py
with open(init_file, "w", encoding="utf-8") as f:
    f.write(init_content)

# Copie de tokenizer.json dans le package
if os.path.exists(tokenizer_file):
    shutil.copy(tokenizer_file, dest_tokenizer_file)
else:
    raise FileNotFoundError(f"Le fichier {tokenizer_file} est introuvable")

# Commande personnalisée pour exécuter les tests après l'installation
class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        # Exécution des tests
        try:
            from structured_gpt2 import tokenizer

            # Test 1: struct_encode
            A = tokenizer.struct_encode(
                [
                    {"role": "user", "content": "structured_gpt2"},
                    {"role": "user", "content": "<|user|>"},
                    {"role": "assistant", "content": "<|assisant|>"}
                ]
            )
            expected_A = [1, 7253, 1526, 66, 74, 461, 21, 1, 49782, 95, 7224, 95, 63, 33, 2, 31, 95, 20301, 419, 95, 33, 3]
            if A != expected_A:
                raise RuntimeError(
                    f"Échec du test struct_encode. Attendu : {expected_A}, Obtenu : {A}. "
                    "Veuillez signaler le problème sur https://github.com/SyntaxError4Life/Structured_GPT-2_tokenizer"
                )

            # Test 2: struct_decode
            B = tokenizer.struct_decode([1, 0, 3])
            expected_B = [{"role": "user", "content": ""}]
            if B != expected_B:
                raise RuntimeError(
                    f"Échec du test struct_decode. Attendu : {expected_B}, Obtenu : {B}. "
                    "Veuillez signaler le problème sur https://github.com/SyntaxError4Life/Structured_GPT-2_tokenizer"
                )

            print("Tous les tests ont réussi.")
        except Exception as e:
            raise RuntimeError(
                f"Échec des tests post-installation : {str(e)}. "
                "Veuillez signaler le problème sur https://github.com/SyntaxError4Life/Structured_GPT-2_tokenizer"
            )

# Configuration du package
setup(
    name=package_name,
    version="0.1.0",
    description="Tokenizer structuré basé sur GPT-2",
    author="SyntaxError4Life",
    packages=[package_name],
    package_data={package_name: ["tokenizer.json"]},
    include_package_data=True,
    install_requires=["tokenizers==0.21.1"],
    cmdclass={
        "install": CustomInstallCommand,
    },
)
