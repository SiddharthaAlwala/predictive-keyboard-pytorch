import re
from pathlib import Path
from typing import List

_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+|[.,!?;:()\"-]")

def load_text(path: str | Path) -> str:
    path = Path(path)
    return path.read_text(encoding='utf-8')

def normalize(text: str)-> str:
    #lowercase for consistency 
    return text.lower()

def tokenize(text: str)-> List[str]:
    #regex-based tokenization
    return _WORD_RE.findall(text)

