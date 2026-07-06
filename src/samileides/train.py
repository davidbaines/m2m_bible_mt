"""Training entry point (spec.md, "Training").

    uv run python -m samileides.train --config configs/experiments/pilot.yaml

HF ``Seq2SeqTrainer`` on a randomly-initialised MarianMT model, bf16, label
smoothing, inverse-sqrt schedule. The tokeniser is trained here on the training
split only and saved beside the checkpoint so generation reuses it. ClearML
logging is opt-in (``--clearml``) and degrades gracefully if unavailable.

Two sanity modes back the verification plan (spec.md #4):
  --overfit N   train on N pairs only; final loss must fall near zero.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import ExperimentConfig
from .data import repo_root
from .data_pipeline import prepare
from .dataset import Collator, PairDataset
from .model import build_model
from .preprocess import SRC_COLUMN, TGT_COLUMN
from .splits import assert_no_leakage
from .tokenizer import load_tokenizer, train_tokenizer


def _maybe_clearml(cfg: ExperimentConfig, enable: bool, remote_queue: str | None):
    """Init a ClearML task; optionally enqueue it to run on a remote agent.

    With ``remote_queue`` set, ``execute_remotely`` captures this repo's git
    commit, the entry point and args, enqueues the task, and stops the local
    process. A worker on that queue then reruns the same command. Returns the
    Task (or None if ClearML is unavailable and not required for remote).
    """
    if not enable and not remote_queue:
        return None
    try:
        from clearml import Task
    except ImportError:
        if remote_queue:
            raise SystemExit("clearml is required for --remote-queue; install the train extra")
        print("clearml not installed; skipping experiment tracking")
        return None
    task = Task.init(project_name="samileides", task_name=cfg.name)
    if remote_queue:
        # Enqueue and exit locally; the agent reruns the whole script remotely.
        task.execute_remotely(queue_name=remote_queue, exit_process=True)
    return task


def _upload_artifacts(output: Path) -> None:
    """If running under a ClearML task, upload the output dir for retrieval."""
    try:
        from clearml import Task
    except ImportError:
        return
    task = Task.current_task()
    if task is None:
        return
    # Upload the whole run directory (model, tokeniser, config, summary, and any
    # generated artefacts) as one zipped artifact, plus register the weights as
    # a model so it is easy to find in the UI.
    task.upload_artifact("run", artifact_object=str(output))
    print(f"  uploaded run artifacts to ClearML task {task.id}")


def train_tokenizer_for(cfg: ExperimentConfig, train_pairs, output: Path):
    """Train (or reuse) the SentencePiece model on the training split only."""
    tok_dir = output / "tokenizer"
    model_path = tok_dir / "spm.model"
    tags = sorted({s.split(" ", 1)[0] for s in train_pairs[SRC_COLUMN]})
    corpus = train_pairs[SRC_COLUMN].tolist() + train_pairs[TGT_COLUMN].tolist()
    train_tokenizer(
        corpus,
        tok_dir / "spm",
        tags=tags,
        vocab_size=cfg.tokenizer.vocab_size,
        model_type=cfg.tokenizer.type,
    )
    return load_tokenizer(model_path)


def build_training_args(cfg: ExperimentConfig, output: Path, args):
    from transformers import Seq2SeqTrainingArguments

    per_device = cfg.training.per_device_batch_size or 16
    max_steps = args.max_steps if args.max_steps else cfg.training.max_steps
    eval_steps = min(cfg.training.eval_every_steps, max_steps)
    # Label smoothing floors the loss well above zero, which hides whether the
    # loop is actually memorising the overfit subset; turn it off there so the
    # near-zero-loss check (spec.md #4) is meaningful.
    label_smoothing = 0.0 if args.overfit else cfg.model.label_smoothing
    return Seq2SeqTrainingArguments(
        output_dir=str(output),
        max_steps=max_steps,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=cfg.training.gradient_accumulation,
        learning_rate=cfg.training.lr,
        warmup_steps=cfg.training.warmup_steps,
        lr_scheduler_type=cfg.training.lr_scheduler,
        max_grad_norm=cfg.training.max_grad_norm,
        label_smoothing_factor=label_smoothing,
        bf16=cfg.training.bf16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(1, min(50, max_steps // 10)),
        seed=cfg.training.seed,
        report_to=["clearml"] if args.clearml else [],
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        predict_with_generate=False,
    )


def run(args) -> None:
    cfg = ExperimentConfig.load(args.config)
    output = Path(args.output_dir) if args.output_dir else repo_root() / "checkpoints" / cfg.name
    output.mkdir(parents=True, exist_ok=True)

    _maybe_clearml(cfg, args.clearml, args.remote_queue)

    print(f"Preparing data for '{cfg.name}' ...")
    data = prepare(cfg)
    assert_no_leakage(data.splits, data.holdouts)
    print(
        f"  pairs: train={len(data.train_pairs)} valid={len(data.valid_pairs)} "
        f"test={len(data.test_pairs)}"
    )

    sp = train_tokenizer_for(cfg, data.train_pairs, output)
    print(f"  tokenizer: {sp.get_piece_size()} pieces "
          f"(unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()} pad={sp.pad_id()})")

    encode = lambda s: sp.encode(s, out_type=int)
    from .preprocess import length_filter

    train_pairs, train_stats = length_filter(
        data.train_pairs, encode, cfg.data.max_len, cfg.data.max_ratio
    )
    valid_pairs, _ = length_filter(
        data.valid_pairs, encode, cfg.data.max_len, cfg.data.max_ratio
    )
    print(f"  length filter (train): {train_stats}")

    if args.overfit:
        train_pairs = train_pairs.head(args.overfit).reset_index(drop=True)
        valid_pairs = train_pairs  # overfit: eval on the same pairs
        print(f"  OVERFIT mode: {len(train_pairs)} pairs")

    train_ds = PairDataset(train_pairs, sp, cfg.data.max_len)
    valid_ds = PairDataset(valid_pairs, sp, cfg.data.max_len)
    collator = Collator(pad_id=sp.pad_id())

    model = build_model(cfg.model, sp, max_position_embeddings=cfg.data.max_len + 4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {cfg.model.arch} {n_params/1e6:.1f}M params")

    from transformers import EarlyStoppingCallback, Seq2SeqTrainer

    targs = build_training_args(cfg, output, args)
    callbacks = []
    if not args.overfit:
        callbacks.append(EarlyStoppingCallback(cfg.training.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"  final eval: {metrics}")

    trainer.save_model(str(output))
    shutil.copy(cfg.path, output / "config.yaml")
    (output / "train_summary.json").write_text(
        json.dumps(
            {
                "name": cfg.name,
                "eval_loss": metrics.get("eval_loss"),
                "n_params": n_params,
                "train_pairs": len(train_pairs),
                "length_filter": train_stats,
                "overfit": args.overfit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model + tokenizer to {output}")

    if args.generate_after and not args.overfit:
        from types import SimpleNamespace

        from .generate import generate_holdouts

        print("Generating and scoring held-out books ...")
        gargs = SimpleNamespace(beam=0, max_length=0, batch_size=args.gen_batch_size)
        generate_holdouts(output, output / "generated", gargs)

    _upload_artifacts(output)

    if args.overfit:
        loss = metrics.get("eval_loss", float("inf"))
        threshold = args.overfit_threshold
        status = "PASS" if loss < threshold else "FAIL"
        print(f"OVERFIT {status}: eval_loss={loss:.4f} (threshold {threshold})")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the samileides MT model")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--overfit", type=int, default=0, help="train on N pairs only")
    p.add_argument("--overfit-threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--clearml", action="store_true")
    p.add_argument("--remote-queue", default=None,
                   help="enqueue this run on a ClearML queue and exit (e.g. jobs_backlog)")
    p.add_argument("--generate-after", action="store_true",
                   help="after training, generate + score held-out books and upload them")
    p.add_argument("--gen-batch-size", type=int, default=32)
    run(p.parse_args())


if __name__ == "__main__":
    main()
