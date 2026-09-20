from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
from transformers.utils import logging as hf_logging


hf_logging.set_verbosity_error()


FIELD_WEIGHTS = {
    "label": 10.0,
    "defect_type": 3.0,
    "explanation": 1.0,
}


def read_jsonl(path: Path, max_samples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


class ThreeFieldDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        max_samples: int = 0,
        normal_abnormal_ratio: float = 0.0,
        balance_labels: bool = False,
        seed: int = 42,
    ):
        self.rows = read_jsonl(manifest, max_samples=max_samples)
        if balance_labels:
            self.rows = self._balance_labels(self.rows, seed)
        elif normal_abnormal_ratio > 0:
            self.rows = self._resample_by_ratio(self.rows, normal_abnormal_ratio, seed)

    @staticmethod
    def _row_label(row: dict[str, Any]) -> str:
        target = row.get("target") or {}
        label = str(target.get("label") or row.get("label") or "").strip().lower()
        if label in {"good", "normal", "0"}:
            return "normal"
        if label in {"abnormal", "defect", "defective", "anomaly", "1"}:
            return "abnormal"
        return label

    @classmethod
    def _split_labels(cls, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        normal = [row for row in rows if cls._row_label(row) == "normal"]
        abnormal = [row for row in rows if cls._row_label(row) == "abnormal"]
        return normal, abnormal

    @classmethod
    def _balance_labels(cls, rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
        normal, abnormal = cls._split_labels(rows)
        if not normal or not abnormal:
            return rows
        rng = random.Random(seed)
        target = max(len(normal), len(abnormal))
        out = list(normal) + list(abnormal)
        if len(normal) < target:
            out.extend(rng.choices(normal, k=target - len(normal)))
        if len(abnormal) < target:
            out.extend(rng.choices(abnormal, k=target - len(abnormal)))
        rng.shuffle(out)
        return out

    @classmethod
    def _resample_by_ratio(cls, rows: list[dict[str, Any]], ratio: float, seed: int) -> list[dict[str, Any]]:
        normal, abnormal = cls._split_labels(rows)
        if not normal or not abnormal:
            return rows
        target_abnormal = max(1, math.ceil(len(normal) / ratio))
        rng = random.Random(seed)
        if len(abnormal) < target_abnormal:
            abnormal = abnormal + rng.choices(abnormal, k=target_abnormal - len(abnormal))
        elif len(abnormal) > target_abnormal:
            abnormal = rng.sample(abnormal, target_abnormal)
        out = list(normal) + list(abnormal)
        rng.shuffle(out)
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def make_messages(row: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for path in row.get("ref_images", []) or []:
        content.append({"type": "image", "image": str(path)})
    content.append({"type": "image", "image": str(row["image"])})
    content.append({"type": "text", "text": str(row["prompt"])})
    messages = [{"role": "user", "content": content}]
    if include_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": str(row["response"])}]})
    return messages


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    first = needle[0]
    for idx in range(len(haystack) - len(needle) + 1):
        if haystack[idx] == first and haystack[idx : idx + len(needle)] == needle:
            return idx
    return -1


class ThreeFieldCollator:
    def __init__(
        self,
        processor: AutoProcessor,
        max_length: int,
        label_weight: float,
        defect_weight: float,
        explanation_weight: float,
    ):
        self.processor = processor
        self.max_length = max_length
        self.field_weights = {
            "label": label_weight,
            "defect_type": defect_weight,
            "explanation": explanation_weight,
        }

    def apply_template(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_messages = [make_messages(row, include_answer=True) for row in rows]
        prompt_messages = [make_messages(row, include_answer=False) for row in rows]
        full_texts = [self.apply_template(messages, add_generation_prompt=False) for messages in full_messages]
        prompt_texts = [self.apply_template(messages, add_generation_prompt=True) for messages in prompt_messages]

        full_images, full_videos = process_vision_info(full_messages)
        prompt_images, prompt_videos = process_vision_info(prompt_messages)
        common: dict[str, Any] = {"padding": True, "return_tensors": "pt"}
        if self.max_length > 0:
            common.update({"truncation": True, "max_length": self.max_length})
        full_inputs = self.processor(text=full_texts, images=full_images, videos=full_videos, **common)
        prompt_inputs = self.processor(text=prompt_texts, images=prompt_images, videos=prompt_videos, **common)

        labels = full_inputs["input_ids"].clone()
        loss_weights = torch.ones_like(labels, dtype=torch.float32)
        for idx in range(labels.shape[0]):
            prompt_len = int(prompt_inputs["attention_mask"][idx].sum().item())
            labels[idx, :prompt_len] = -100
            self.apply_field_weights(loss_weights[idx], full_inputs["input_ids"][idx], prompt_len, str(rows[idx]["response"]))
        labels[full_inputs["attention_mask"] == 0] = -100
        loss_weights[labels == -100] = 0.0
        full_inputs["labels"] = labels
        full_inputs["loss_weights"] = loss_weights
        return full_inputs

    def apply_field_weights(self, weights: torch.Tensor, input_ids: torch.Tensor, prompt_len: int, answer: str) -> None:
        tokenizer = self.processor.tokenizer
        try:
            encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
        except Exception:
            return
        answer_ids = encoded.get("input_ids", [])
        offsets = encoded.get("offset_mapping") or []
        rel_start = find_subsequence(input_ids.tolist()[prompt_len:], answer_ids)
        if rel_start < 0:
            rel_start = 0
        abs_start = prompt_len + rel_start
        for field, field_weight in self.field_weights.items():
            match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', answer)
            if not match:
                continue
            start_char, end_char = match.span(1)
            for token_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_end <= start_char or tok_start >= end_char:
                    continue
                pos = abs_start + token_idx
                if pos < weights.numel():
                    weights[pos] = max(float(weights[pos]), float(field_weight))


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weights = inputs.pop("loss_weights", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if labels is None:
            loss = outputs.loss
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels)
            valid = (shift_labels != -100).float()
            if loss_weights is not None:
                valid = valid * loss_weights[..., 1:].contiguous().to(token_loss.device)
            loss = (token_loss * valid).sum() / valid.sum().clamp_min(1.0)
        return (loss, outputs) if return_outputs else loss


def load_model(model_path: str, load_in_4bit: bool):
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "device_map": "auto",
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    try:
        return AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen LoRA on the three-field anomaly JSON protocol.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-pixels", type=int, default=147456)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--normal-abnormal-ratio", type=float, default=0.0)
    parser.add_argument("--balance-labels", action="store_true")
    parser.add_argument("--label-loss-weight", type=float, default=10.0)
    parser.add_argument("--defect-loss-weight", type=float, default=3.0)
    parser.add_argument("--explanation-loss-weight", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage] start train_manifest={args.train_manifest} model={args.model} output={args.output_dir}", flush=True)

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        min_pixels=224 * 224,
        max_pixels=args.max_pixels,
    )
    print("[stage] processor_loaded", flush=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_model(args.model, args.load_in_4bit)
    print("[stage] model_loaded", flush=True)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if not args.no_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    print("[stage] lora_attached", flush=True)
    model.print_trainable_parameters()

    train_ds = ThreeFieldDataset(
        Path(args.train_manifest),
        max_samples=args.max_samples,
        normal_abnormal_ratio=args.normal_abnormal_ratio,
        balance_labels=args.balance_labels,
        seed=args.seed,
    )
    print(f"[stage] train_dataset_ready n={len(train_ds)}", flush=True)
    eval_ds = ThreeFieldDataset(Path(args.val_manifest)) if args.val_manifest else None
    if eval_ds is not None:
        print(f"[stage] eval_dataset_ready n={len(eval_ds)}", flush=True)
    collator = ThreeFieldCollator(
        processor=processor,
        max_length=args.max_length,
        label_weight=args.label_loss_weight,
        defect_weight=args.defect_loss_weight,
        explanation_weight=args.explanation_loss_weight,
    )
    print("[stage] collator_ready", flush=True)

    train_args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "bf16": True,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "report_to": [],
        "optim": "adamw_torch",
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "max_grad_norm": 1.0,
        "seed": args.seed,
    }
    if eval_ds is not None:
        train_args_kwargs["per_device_eval_batch_size"] = max(1, args.batch_size)
    train_args = TrainingArguments(**train_args_kwargs)
    trainer = WeightedTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    print("[stage] trainer_ready", flush=True)
    print("[stage] train_begin", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    print("[stage] train_end", flush=True)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    (output_dir / "TRAINING_DONE").write_text("done\n", encoding="utf-8")
    print("[stage] saved", flush=True)


if __name__ == "__main__":
    main()
