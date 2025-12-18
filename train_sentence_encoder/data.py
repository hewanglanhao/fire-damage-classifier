import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = {
    "pad": "[PAD]",
    "unk": "[UNK]",
    "cls": "[CLS]",
    "sep": "[SEP]",
    "mask": "[MASK]",
}


class SentenceTokenizer:
    def __init__(self, vocab_size: int = 20000, max_length: int = 64, lowercase: bool = True):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.lowercase = lowercase
        self.token_to_id = {}
        self.id_to_token = {}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[SPECIAL_TOKENS["pad"]]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[SPECIAL_TOKENS["unk"]]

    @property
    def cls_id(self) -> int:
        return self.token_to_id[SPECIAL_TOKENS["cls"]]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[SPECIAL_TOKENS["sep"]]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[SPECIAL_TOKENS["mask"]]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def _basic_tokenize(self, text: str) -> List[str]:
        if text is None:
            return []
        text = text.strip()
        if self.lowercase:
            text = text.lower()
        return text.split()

    def build(self, texts: List[str]) -> None:
        counts = Counter()
        for text in texts:
            counts.update(self._basic_tokenize(text))

        specials = list(SPECIAL_TOKENS.values())
        self.token_to_id = {tok: idx for idx, tok in enumerate(specials)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

        limit = self.vocab_size - len(specials)
        for idx, (token, _) in enumerate(counts.most_common(limit), start=len(specials)):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._basic_tokenize(text)
        trimmed = tokens[: self.max_length - 2]
        token_ids = [self.cls_id] + [
            self.token_to_id.get(tok, self.unk_id) for tok in trimmed
        ] + [self.sep_id]

        # Pad
        pad_len = self.max_length - len(token_ids)
        if pad_len > 0:
            token_ids += [self.pad_id] * pad_len
        else:
            token_ids = token_ids[: self.max_length]

        attention_mask = [1 if tid != self.pad_id else 0 for tid in token_ids]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab_size": self.vocab_size,
                    "max_length": self.max_length,
                    "lowercase": self.lowercase,
                    "token_to_id": self.token_to_id,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.max_length = data["max_length"]
        self.lowercase = data.get("lowercase", True)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}


def load_texts(jsonl_path: str, field: str) -> List[str]:
    valid_fields = {"coarse": "inference_coarse", "fine": "inference_fine"}
    if field not in valid_fields:
        raise ValueError(f"field must be one of {list(valid_fields)}")

    target_key = valid_fields[field]
    texts: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = item.get(target_key, "")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


def mask_input_ids(input_ids: torch.Tensor, tokenizer: SentenceTokenizer, mlm_prob: float = 0.15):
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_prob)

    special_ids = {tokenizer.pad_id, tokenizer.cls_id, tokenizer.sep_id}
    for special_id in special_ids:
        probability_matrix = probability_matrix * (input_ids != special_id)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # 80% -> [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_id

    # 10% -> random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # 10% -> keep original (already handled by leaving masked_indices but not replaced/random)
    return input_ids, labels


class SentenceDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: SentenceTokenizer,
        mlm_prob: float = 0.15,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        input_ids, attention_mask = self.tokenizer.encode(text)
        mlm_input_ids, labels = mask_input_ids(input_ids.clone(), self.tokenizer, self.mlm_prob)
        return {
            "input_ids": mlm_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
