"""
Symbol Core: Semantic Compression & Token Reduction Engine
Prototype v0.1
"""

import json
import re

DEFAULT_DICTIONARY = {
    # Control flow & logic
    "function": "ƒ",
    "return": "→",
    "if": "?",
    "else": "‽",
    "for": "∀",
    "while": "↻",
    "class": "ℂ",
    "import": "📦",
    "export": "📤",
    "async": "⚡",
    "await": "⏳",

    # Common programmatic tokens / concepts
    "error": "❌",
    "success": "✅",
    "request": "🌐",
    "response": "📨",
    "database": "🗄️",
    "query": "🔍",
    "user": "👤",
    "config": "⚙️"
}

class SymbolEncoder:
    def __init__(self, dictionary=None):
        self.dict = dictionary or DEFAULT_DICTIONARY
        # Sort keys by length descending to replace longer phrases first
        self.sorted_keys = sorted(self.dict.keys(), key=len, reverse=True)

    def compress(self, text: str) -> str:
        """Compresses verbose text/code into high-density symbol representation."""
        compressed = text
        for key in self.sorted_keys:
            symbol = self.dict[key]
            # Word boundary matching where appropriate, or direct replacement
            pattern = r'\b' + re.escape(key) + r'\b'
            compressed = re.sub(pattern, symbol, compressed)
        return compressed

    def decompress(self, text: str) -> str:
        """Expands symbol representation back to readable text/code."""
        decompressed = text
        inv_dict = {v: k for k, v in self.dict.items()}
        sorted_symbols = sorted(inv_dict.keys(), key=len, reverse=True)

        for symbol in sorted_symbols:
            word = inv_dict[symbol]
            decompressed = decompressed.replace(symbol, word)
        return decompressed

if __name__ == "__main__":
    encoder = SymbolEncoder()
    sample_code = "async function requestUser(user) { if (user) { return response; } else { error; } }"

    compressed = encoder.compress(sample_code)
    decompressed = encoder.decompress(compressed)

    print("Original Length:", len(sample_code))
    print("Compressed Length:", len(compressed))
    print("Compressed Output:", compressed)
    print("Decompressed Match:", decompressed == sample_code)
