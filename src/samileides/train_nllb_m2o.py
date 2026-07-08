"""One many-to-one NLLB run: train NT-only, generate withheld OT books, score.

    uv run python -m samileides.train_nllb_m2o --config configs/experiments/m2o/ton.yaml --init relative

Trains a small NLLB fine-tune from K same-family sources into one target, adds
the target token per the chosen init method, then generates the withheld books
from each source and scores against the target reference. The model is not
saved (the deliverable is the score table); results append to a CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data import load_verses, repo_root
from .dataset import Collator
from .evaluate import score
from .nllb import NllbDataset
from .nllb_m2o import (add_target_token, build_m2o_pairs, generate_book,
                       target_token_for, test_vrefs, usable)

INIT_FROM = {"relative": "relative", "scratch": None, "same_script": "script_anchor"}


def append_results(res_path: Path, rows: list[dict]) -> None:
    """Append result rows to a shared CSV, refusing to misalign an old schema.

    The header is written only for a new file; for an existing file the
    column set must match exactly, otherwise DictWriter would silently write
    shifted columns under the old header.
    """
    res_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    if res_path.exists():
        first = res_path.read_text(encoding="utf-8").splitlines()
        if first and first[0].split(",") != fieldnames:
            raise SystemExit(
                f"{res_path} has columns [{first[0]}] but this run produces "
                f"{fieldnames}; appending would silently misalign. "
                "Point --results at a fresh file."
            )
        header = not first
    else:
        header = True
    with res_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if header:
            w.writeheader()
        w.writerows(rows)


def run(args) -> None:
    import numpy as np
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, EarlyStoppingCallback,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    from .nllb_m2o import build_valid_pairs

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tgt = cfg["target"]
    sources = [(s["tid"], s["flores"]) for s in cfg["sources"]]
    same_script = sum(1 for s in cfg["sources"] if s["flores"].split("_")[1] == tgt["script"])

    # load all needed columns once
    tids = [tgt["tid"]] + [s["tid"] for s in cfg["sources"]]
    verses = load_verses(tids)

    tok = AutoTokenizer.from_pretrained(cfg["pretrained"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["pretrained"], dtype=torch.bfloat16)
    if cfg["training"].get("gradient_checkpointing", True):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # decide the target token + init (token choice shared with publish_nllb)
    target_token = target_token_for(cfg, args.init, tok.get_vocab())
    if args.init == "existing":
        tgt_id = tok.convert_tokens_to_ids(target_token)
    else:
        init_key = INIT_FROM[args.init]
        init_from = cfg[init_key] if init_key else None
        tgt_id = add_target_token(tok, model, target_token, init_from)

    # generation must force the target token first (used by eval + test generation)
    model.generation_config.forced_bos_token_id = tgt_id

    # fixed 250-verse NT validation set, shared across all experiments
    val_vrefs = [l.strip() for l in Path(args.valid_vrefs).read_text().splitlines() if l.strip()]
    train_pairs = build_m2o_pairs(tgt["tid"], sources, verses, target_token, exclude_vrefs=val_vrefs)
    val_pairs = build_valid_pairs(tgt["tid"], sources, verses, target_token, val_vrefs)
    ml = cfg["training"].get("max_len", 128)
    ds = NllbDataset(train_pairs, tok, ml)
    val_ds = NllbDataset(val_pairs, tok, ml)
    collator = Collator(pad_id=tok.pad_token_id, decoder_start_id=tok.eos_token_id)
    print(f"{cfg['name']} init={args.init}: {len(train_pairs)} NT train pairs, "
          f"{len(val_pairs)} val verses, target token {target_token!r}, "
          f"same-script sources {same_script}/{len(sources)}")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        hyps = tok.batch_decode(preds, skip_special_tokens=True)
        refs = tok.batch_decode(labels, skip_special_tokens=True)
        return {"chrf3": score(hyps, refs)["chrF3"]}

    tr = cfg["training"]
    lr = args.lr if args.lr is not None else float(tr.get("lr", 3e-5))
    out = Path(args.tmp_dir) / f"{cfg['name']}_{args.init}"
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out), max_steps=args.steps or tr.get("steps", 20000),
        per_device_train_batch_size=tr.get("batch", 16),
        per_device_eval_batch_size=tr.get("eval_batch", 32),
        gradient_accumulation_steps=tr.get("grad_accum", 1),
        learning_rate=lr, warmup_steps=tr.get("warmup", 100),
        lr_scheduler_type="inverse_sqrt", bf16=True, label_smoothing_factor=0.1,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_chrf3", greater_is_better=True,
        predict_with_generate=True, generation_num_beams=args.val_beam, generation_max_length=ml,
        logging_steps=200, report_to=[], remove_unused_columns=False, dataloader_num_workers=2,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience,
                                        early_stopping_threshold=args.min_delta)],
    )
    trainer.train()
    curve = [(h["step"], h["eval_chrf3"]) for h in trainer.state.log_history if "eval_chrf3" in h]
    best = max(curve, key=lambda x: x[1]) if curve else (0, 0.0)
    print(f"VAL chrF3 curve: {curve}")
    print(f"BEST val chrF3 {best[1]} at step {best[0]} / max {targs.max_steps}; "
          f"runtime {trainer.state.log_history[-1].get('train_runtime','?')}s")

    # generate the withheld books from each source, score vs the target
    model = model.to(device).eval()
    gen = cfg["generate"]
    books = test_vrefs(verses, gen)
    results_rows = []
    for book, vrefs in books.items():
        refs_all = {v: (verses.at[v, tgt["tid"]] or "") for v in vrefs}
        best = None
        per_source = {}
        for s in cfg["sources"]:
            idx = [v for v in vrefs if usable(verses.at[v, s["tid"]]) and usable(refs_all[v])]
            if not idx:
                continue
            srcs = [verses.at[v, s["tid"]] for v in idx]
            refs = [refs_all[v] for v in idx]
            hyps = generate_book(model, tok, device, srcs, s["flores"], tgt_id,
                                gen.get("beam", 5), gen.get("max_length", 128))
            m = score(hyps, refs)
            per_source[s["code"]] = m["chrF3"]
            if best is None or m["chrF3"] > best[1]:
                best = (s["code"], m["chrF3"], m["spBLEU"], len(idx))
        if best:
            results_rows.append({
                "target": tgt["code"], "init": args.init, "lr": lr, "book": book,
                "same_script_sources": same_script, "n_sources": len(sources),
                "best_source": best[0], "best_chrF3": best[1], "best_spBLEU": best[2],
                "mean_chrF3": round(float(np.mean(list(per_source.values()))), 2),
                "verses": best[3], "per_source": str(per_source),
            })
            print(f"  {book}: best {best[0]} chrF3={best[1]} (mean {results_rows[-1]['mean_chrF3']})")

    # append to the shared results CSV
    res_path = Path(args.results)
    append_results(res_path, results_rows)
    print(f"Appended {len(results_rows)} rows to {res_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="One many-to-one NLLB run")
    p.add_argument("--config", required=True)
    p.add_argument("--init", required=True, choices=["relative", "scratch", "same_script", "existing"])
    p.add_argument("--results", default=str(repo_root() / "experiments" / "m2o-results.csv"))
    p.add_argument("--tmp-dir", default="/tmp/claude-1000/-home-david-Documents-Github/95a24bdc-9442-45d9-9e43-b9fb7fe88cf1/scratchpad/m2o")
    p.add_argument("--steps", type=int, default=None, help="override max training steps")
    p.add_argument("--lr", type=float, default=None, help="override the config learning rate")
    p.add_argument("--eval-steps", type=int, default=1000, help="generate+score the val set every N steps")
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience (evals)")
    p.add_argument("--min-delta", type=float, default=0.2,
                   help="min chrF3 gain to count as improvement (avoids noise-driven overruns)")
    p.add_argument("--val-beam", type=int, default=5,
                   help="beam for the validation generation probe (matches the test beam)")
    p.add_argument("--valid-vrefs", default=str(repo_root() / "experiments" / "m2o-valid-vrefs.txt"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
