"""Proxy benchmark tasks for the active segment-memory language-model wrapper."""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+|[^\w\s]")


def _lookup_token_id(token: str, token_to_id: Dict[str, int]) -> int:
    if token in token_to_id:
        return token_to_id[token]
    lowered = token.lower()
    if lowered in token_to_id:
        return token_to_id[lowered]
    return token_to_id.get("<unk>", 1)


def _tokenize_to_ids(text: str, token_to_id: Dict[str, int]) -> List[int]:
    tokens = _TOKEN_PATTERN.findall(text)
    if not tokens:
        return [token_to_id.get("<unk>", 1)]
    return [_lookup_token_id(token, token_to_id) for token in tokens]


def _require_language_model(model) -> Tuple[int, Dict[str, int]]:
    if not (hasattr(model, "embedding") and hasattr(model, "output_layer")):
        raise ValueError(
            "Proxy benchmark evaluation requires a language-model style wrapper with "
            "`embedding` and `output_layer`."
        )
    token_to_id = getattr(model, "token_to_id", None)
    if not token_to_id:
        raise ValueError(
            "Proxy benchmark evaluation requires the language-model wrapper to expose "
            "`token_to_id` so evaluation uses the same vocabulary as training."
        )
    return int(model.output_layer.out_features), token_to_id


def _resolve_special_token_id(model, token_to_id: Dict[str, int], token_name: str, fallback: int) -> int:
    attr_name = f"{token_name.strip('<>')}_token_id"
    token_id = getattr(model, attr_name, None)
    if token_id is not None:
        return int(token_id)
    return int(token_to_id.get(token_name, fallback))


def _sequence_logprob(
    model,
    prompt_tokens: List[int],
    continuation_tokens: List[int],
    device: torch.device,
) -> float:
    if not continuation_tokens:
        return float("-inf")

    _, token_to_id = _require_language_model(model)
    bos_id = _resolve_special_token_id(
        model,
        token_to_id,
        "<bos>",
        token_to_id.get("<unk>", 1),
    )

    full_tokens = [bos_id] + list(prompt_tokens) + list(continuation_tokens)
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    input_ids = torch.tensor(input_tokens, dtype=torch.long, device=device)
    if getattr(model, "batch_first", False):
        input_ids = input_ids.unsqueeze(0)
    else:
        input_ids = input_ids.unsqueeze(1)

    logits, _, _, _ = model(input_ids, return_all_outputs=True, return_hidden_states=False)
    step_logits = logits[0] if getattr(model, "batch_first", False) else logits[:, 0, :]
    target_ids = torch.tensor(target_tokens, dtype=torch.long, device=device)
    token_log_probs = F.log_softmax(step_logits, dim=-1).gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    continuation_start = len(prompt_tokens)
    continuation_log_probs = token_log_probs[continuation_start:]
    if continuation_log_probs.numel() == 0:
        return float("-inf")
    return continuation_log_probs.mean().item()


def _score_choice(model, prompt: str, choice: str, token_to_id: Dict[str, int], device: torch.device) -> float:
    prompt_tokens = _tokenize_to_ids(prompt, token_to_id)
    choice_tokens = _tokenize_to_ids(choice, token_to_id)
    return _sequence_logprob(model, prompt_tokens, choice_tokens, device)


class NeedleInHaystackBenchmarkDataset(Dataset):
    def __init__(
        self,
        context_len: int = 4096,
        needle_depth: float = 0.5,
        num_samples: int = 100,
        token_to_id: Dict[str, int] | None = None,
    ):
        self.context_len = context_len
        self.needle_depth = needle_depth
        self.num_samples = num_samples
        self.token_to_id = dict(token_to_id or {})
        self.data, self.targets, self.needle_values = self._generate_data()

    def _generate_data(self) -> Tuple[List[torch.Tensor], List[str], List[str]]:
        needles = [
            ("The magic number is 42", "42"),
            ("The secret code is X7Z9", "X7Z9"),
            ("The answer is Paris", "Paris"),
            ("The special value is 999", "999"),
        ]
        special_ids = {
            self.token_to_id.get("<pad>", -1),
            self.token_to_id.get("<unk>", -1),
            self.token_to_id.get("<bos>", -1),
            self.token_to_id.get("<eos>", -1),
        }
        haystack_tokens = [
            token_id
            for _, token_id in sorted(self.token_to_id.items(), key=lambda item: item[1])
            if token_id not in special_ids
        ][:2048]
        if not haystack_tokens:
            haystack_tokens = [self.token_to_id.get("<unk>", 1)]

        data, targets, needle_values = [], [], []
        for i in range(self.num_samples):
            needle_text, answer = needles[i % len(needles)]
            needle_tokens = _tokenize_to_ids(needle_text, self.token_to_id)
            needle_pos = int(self.context_len * self.needle_depth)
            needle_pos = max(len(needle_tokens), min(self.context_len - len(needle_tokens) - 1, needle_pos))

            context_tokens = []
            for j in range(self.context_len):
                if needle_pos <= j < needle_pos + len(needle_tokens):
                    context_tokens.append(needle_tokens[j - needle_pos])
                else:
                    context_tokens.append(haystack_tokens[(i * 131 + j) % len(haystack_tokens)])

            data.append(torch.tensor(context_tokens, dtype=torch.long))
            targets.append(answer)
            needle_values.append(needle_text)

        return data, targets, needle_values

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.needle_values[idx]


@torch.no_grad()
def evaluate_needle_in_haystack(
    model,
    device: torch.device,
    context_len: int = 4096,
    needle_depth: float = 0.5,
    batch_size: int = 4,
    vocab_size: int = 10000,
    **kwargs,
) -> Dict[str, Any]:
    model.eval()
    _, token_to_id = _require_language_model(model)
    dataset = NeedleInHaystackBenchmarkDataset(
        context_len=context_len,
        needle_depth=needle_depth,
        num_samples=100,
        token_to_id=token_to_id,
    )

    answer_candidates = ["42", "X7Z9", "Paris", "999"]
    correct = 0
    total = 0
    details = []

    for contexts, targets, needle_values in DataLoader(dataset, batch_size=1, shuffle=False):
        context_tokens = contexts[0].tolist()
        target = targets[0]
        needle_text = needle_values[0]

        prompt_tokens = context_tokens + _tokenize_to_ids("question: what is the hidden value ? answer:", token_to_id)
        scores = {
            candidate: _sequence_logprob(
                model,
                prompt_tokens=prompt_tokens,
                continuation_tokens=_tokenize_to_ids(candidate, token_to_id),
                device=device,
            )
            for candidate in answer_candidates
        }

        prediction = max(scores, key=scores.get)
        is_correct = prediction == target
        correct += int(is_correct)
        total += 1
        details.append(
            {
                "needle": needle_text,
                "target": target,
                "prediction": prediction,
                "scores": scores,
                "correct": is_correct,
            }
        )

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "retrieval_accuracy": accuracy,
        "value_accuracy": accuracy,
        "context_len": context_len,
        "needle_depth": needle_depth,
        "total_samples": total,
        "details": details,
    }


class PIQABenchmarkDataset(Dataset):
    def __init__(self, data_path: str, split: str = "val"):
        self.data_path = Path(data_path)
        self.split = split
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        path = self.data_path / f"piqa_{self.split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"PIQA data file not found: {path}")

        data = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["goal"], item["sol1"], item["sol2"], item.get("label", 0)


def evaluate_piqa_benchmark(
    model,
    device: torch.device,
    data_path: str = "./data/piqa",
    batch_size: int = 16,
    hidden_size: int = 128,
) -> Dict[str, Any]:
    model.eval()
    _, token_to_id = _require_language_model(model)
    dataset = PIQABenchmarkDataset(data_path, split="val")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    for questions, option1s, option2s, labels in dataloader:
        for q, o1, o2, label in zip(questions, option1s, option2s, labels):
            prompt = f"question: {q}\nanswer:"
            score1 = _score_choice(model, prompt, o1, token_to_id, device)
            score2 = _score_choice(model, prompt, o2, token_to_id, device)
            pred = 0 if score1 >= score2 else 1
            correct += int(pred == int(label))
            total += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "total_samples": total,
        "is_dummy_data": False,
    }


class HellaSwagBenchmarkDataset(Dataset):
    def __init__(self, data_path: str, split: str = "val"):
        self.data_path = Path(data_path)
        self.split = split
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        path = self.data_path / f"hellaswag_{self.split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"HellaSwag data file not found: {path}")

        data = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["ctx"], item["endings"], item.get("label", 0)


def evaluate_hellaswag_benchmark(
    model,
    device: torch.device,
    data_path: str = "./data/hellaswag",
    batch_size: int = 16,
    hidden_size: int = 128,
) -> Dict[str, Any]:
    model.eval()
    _, token_to_id = _require_language_model(model)
    dataset = HellaSwagBenchmarkDataset(data_path, split="val")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    for ctxs, endings_list, labels in dataloader:
        for ctx, endings, label in zip(ctxs, endings_list, labels):
            prompt = f"context: {ctx}\nending:"
            scores = [_score_choice(model, prompt, ending, token_to_id, device) for ending in endings]
            pred = max(range(len(scores)), key=lambda idx: scores[idx]) if scores else 0
            correct += int(pred == int(label))
            total += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "total_samples": total,
        "is_dummy_data": False,
    }


def run_proxy_benchmark_suite(
    model,
    device: torch.device,
    data_path: str = "./data",
    hidden_size: int = 128,
) -> Dict[str, Any]:
    def _run_task(fn, *args, **kwargs) -> Dict[str, Any]:
        try:
            result = fn(*args, **kwargs)
            result.setdefault("status", "ok")
            return result
        except FileNotFoundError as exc:
            return {"status": "missing_data", "error": str(exc)}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    results: Dict[str, Any] = {}

    for context_len in [1024, 2048, 4096]:
        for depth in [0.0, 0.5, 1.0]:
            key = f"niah_{context_len // 1024}K_depth{int(depth * 10)}"
            results[key] = _run_task(
                evaluate_needle_in_haystack,
                model,
                device,
                context_len=context_len,
                needle_depth=depth,
            )

    results["piqa"] = _run_task(
        evaluate_piqa_benchmark,
        model,
        device,
        data_path=f"{data_path}/piqa",
        hidden_size=hidden_size,
    )

    results["hellaswag"] = _run_task(
        evaluate_hellaswag_benchmark,
        model,
        device,
        data_path=f"{data_path}/hellaswag",
        hidden_size=hidden_size,
    )

    return results


__all__ = [
    "NeedleInHaystackBenchmarkDataset",
    "PIQABenchmarkDataset",
    "HellaSwagBenchmarkDataset",
    "evaluate_needle_in_haystack",
    "evaluate_piqa_benchmark",
    "evaluate_hellaswag_benchmark",
    "run_proxy_benchmark_suite",
]
