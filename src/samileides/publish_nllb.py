"""Publish an NLLB many-to-one fine-tune to the Hugging Face Hub.

    uv run python -m samileides.publish_nllb --config configs/experiments/m2o/ton.yaml \
        --init scratch --checkpoint runs/m2o_winners/m2o_ton_scratch --dry-run

Sibling of ``samileides.publish`` (which handles the from-scratch Marian runs)
for the NLLB m2o track. The same two gates apply, adapted to this pipeline:

1. **Quality**: every generated test book must beat the source-copy baseline
   (chrF3 of each source language's own text against the target reference).
   In the m2o setting the sources are the target's relatives, so this is also
   the "other-language" floor of the main pipeline — one baseline serves both.
   Scores come from the matrix results CSV; baselines are computed here from
   the corpus.
2. **Licence**: every training translation must be shareable. ``by-nc``
   sources are accepted, because the constraint is binding anyway: the base
   model (NLLB-200) is CC-BY-NC-4.0, so any fine-tune of it is non-commercial
   regardless of the data. The staged licence always carries the NC clause.

The tokenizer was not saved by training (deliberately — the matrix discarded
models); it is reconstructed here exactly as training built it:
``add_target_token`` appends the new code at id ``len(tokenizer)``, which is
deterministic given the same base tokenizer.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from .data import load_verses, repo_root
from .evaluate import score
from .licensing import is_shareable, licence_of, model_licence_for
from .nllb_m2o import test_vrefs

NLLB_BASE_LICENCE = "cc-by-nc-4.0"  # facebook/nllb-200-* are CC-BY-NC-4.0


def default_repo_id(experiment: str) -> str:
    return f"DavidCBaines/ebible_m2o-nllb600m-{experiment.removeprefix('m2o_').replace('_', '-')}"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
        ).strip()
    except Exception:
        return "unknown"


def nllb_model_licence(data_licences) -> str:
    """The licence a published NLLB fine-tune may carry.

    The data-side licence is computed as usual (ShareAlike propagates), then
    the NC clause is forced because the base model is CC-BY-NC-4.0 — even a
    run trained purely on Public Domain text cannot be more permissive than
    its base weights.
    """
    base = model_licence_for(data_licences, allow_nc=True)
    if base == "cc0-1.0":
        return NLLB_BASE_LICENCE
    if "-nc" not in base:
        base = base.replace("cc-by", "cc-by-nc")
    return base


def target_token_for(cfg: dict, init: str, base_vocab) -> str:
    """The target token the run trained with (mirrors train_nllb_m2o)."""
    if init == "existing":
        return cfg["existing_token"]
    token = cfg["target"]["new_token"]
    if token in base_vocab:
        token = token + "_new"
    return token


def source_copy_baselines(cfg: dict, verses: pd.DataFrame) -> dict[str, dict[str, float]]:
    """chrF3 of each source's own text against the target reference, per book."""
    tgt_tid = cfg["target"]["tid"]
    books = test_vrefs(verses, cfg["generate"])
    out: dict[str, dict[str, float]] = {}
    for book, vrefs in books.items():
        per_source = {}
        for s in cfg["sources"]:
            idx = [v for v in vrefs if verses.at[v, s["tid"]] and verses.at[v, tgt_tid]]
            if not idx:
                continue
            hyps = [verses.at[v, s["tid"]] for v in idx]
            refs = [verses.at[v, tgt_tid] for v in idx]
            per_source[s["code"]] = score(hyps, refs)["chrF3"]
        out[book] = per_source
    return out


def check_gate(rows: pd.DataFrame, baselines: dict[str, dict[str, float]]) -> tuple[bool, list[str]]:
    """Every book's generated best must beat every source's copy baseline."""
    problems = []
    for _, r in rows.iterrows():
        floor_src, floor = max(
            baselines.get(r["book"], {"?": 0.0}).items(), key=lambda kv: kv[1]
        )
        if r["best_chrF3"] <= floor:
            problems.append(
                f"{r['book']}: generated {r['best_chrF3']} does not beat the "
                f"source-copy floor {floor} ({floor_src})"
            )
    return not problems, problems


def matrix_rows(results_csv: Path, target_code: str, init: str) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    rows = df[(df["target"] == target_code) & (df["init"] == init)]
    if rows.empty:
        raise SystemExit(
            f"No rows for target={target_code} init={init} in {results_csv}. "
            "Publish only runs that are in the matrix results."
        )
    return rows.reset_index(drop=True)


def _score_table(rows: pd.DataFrame, baselines: dict[str, dict[str, float]]) -> str:
    out_rows = []
    for _, r in rows.iterrows():
        floor = max(baselines.get(r["book"], {}).values(), default=0.0)
        out_rows.append({
            "book": r["book"], "verses": r["verses"],
            "best source": r["best_source"], "chrF3": r["best_chrF3"],
            "spBLEU": r["best_spBLEU"], "mean chrF3 over sources": r["mean_chrF3"],
            "source-copy floor": round(floor, 2),
        })
    detail = []
    for _, r in rows.iterrows():
        per_source = ast.literal_eval(r["per_source"])
        cells = ", ".join(f"{k} {v}" for k, v in sorted(per_source.items()))
        detail.append(f"- {r['book']}: chrF3 by source — {cells}")
    return (
        pd.DataFrame(out_rows).to_markdown(index=False)
        + "\n\nPer-source detail:\n\n" + "\n".join(detail)
    )


def build_model_card(
    *, repo_id: str, cfg: dict, init: str, target_token: str,
    rows: pd.DataFrame, baselines: dict, licences: dict[str, str],
    model_licence: str, best_val: str, commit: str,
) -> str:
    tgt = cfg["target"]
    languages = sorted({tgt["code"], *(s["code"] for s in cfg["sources"])})
    lang_lines = "\n".join(f"  - {c}" for c in languages)
    src_lines = "\n".join(
        f"| `{s['tid']}` | {s['code']} | `{s['flores']}` | {licences[s['tid']]} |"
        for s in cfg["sources"]
    )
    example_src = cfg["sources"][0]
    header = (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: translation\n"
        f"license: {model_licence}\n"
        "base_model: facebook/nllb-200-distilled-600M\n"
        "language:\n"
        f"{lang_lines}\n"
        "tags:\n  - translation\n  - bible\n  - nllb\n  - low-resource\n"
        "---\n"
    )
    body = f"""# {repo_id.split('/')[-1]}

NLLB-200-distilled-600M fine-tuned to translate Bible verses **into
{tgt['code']}** from four related source languages. Part of a series testing
whether a pretrained multilingual model can draft Old Testament books for a
language it has only seen the New Testament of — the practical "no OT exists
yet" scenario. Project: reproduction and extension of Sami Liedes' 2018
closed-text Bible translation experiment.

Training used **New Testament verses only** ({tgt['code']} target, whole OT
withheld); the scores below are on withheld OT material (Ruth, Jonah,
Genesis 1) the model never saw in the target language. Target token
`{target_token}` ({init} init — the token-init comparison in the experiment
showed the init method makes no measurable difference at an adequate
learning rate).

## How to use

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

tokenizer.src_lang = "{example_src['flores']}"  # any source language's FLORES code
batch = tokenizer(["<a {example_src['code']} verse>"], return_tensors="pt")
out = model.generate(
    **batch,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("{target_token}"),
    num_beams=5, max_length=128,
)
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
```

The `forced_bos_token_id` is also baked into `generation_config.json`, so
omitting it works too.

## Evaluation (withheld OT books, never seen in {tgt['code']})

chrF3, sacreBLEU conventions. The **source-copy floor** is the chrF3 of the
best source language's own text against the {tgt['code']} reference — the
score you would get by simply copying the closest relative. The model must
beat it for the run to be publishable. Validation (250 NT verses): best
chrF3 {best_val}.

{_score_table(rows, baselines)}

## Training data and licensing

Fine-tuned on eBible-corpus translations. The base model (NLLB-200) is
CC-BY-NC-4.0, so this model is **non-commercial** regardless of the data
licences below; ShareAlike sources additionally propagate SA. Released under
`{model_licence}`.

| translation | language | FLORES code | licence |
|---|---|---|---|
| `{tgt['tid']}` | {tgt['code']} (target) | — | {licences[tgt['tid']]} |
{src_lines}

## Reproducibility

- Experiment: `{cfg['name']}`, init `{init}`, lr 3e-4, max 8000 steps,
  generation-based early stopping (chrF3 on a fixed 250-verse NT set,
  patience 3, min-delta 0.2).
- Git commit: `{commit}`
- Code, configs and the full 15-run comparison: the project repository and
  the companion results dataset.

## Acknowledgement

This series extends the closed-text Bible translation experiment described by
Sami Liedes (2018). The work is independent and not endorsed by him.
"""
    return header + "\n" + body


def assemble(args) -> dict:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint)
    staging = Path(args.staging_dir) if args.staging_dir else checkpoint.parent / f"{cfg['name']}_{args.init}_hf_staging"
    repo_id = args.repo or default_repo_id(cfg["name"])

    # licence gate (NC is acceptable: the NLLB base forces NC anyway)
    tids = [cfg["target"]["tid"]] + [s["tid"] for s in cfg["sources"]]
    licences = licence_of(tids)
    offenders = {t: l for t, l in licences.items() if not is_shareable(l, allow_nc=True)}
    if offenders:
        raise SystemExit(f"Licence gate FAILED: non-shareable sources {offenders}")
    model_licence = nllb_model_licence(licences.values())

    # quality gate against freshly computed source-copy floors
    rows = matrix_rows(Path(args.results), cfg["target"]["code"], args.init)
    verses = load_verses(tids)
    baselines = source_copy_baselines(cfg, verses)
    passed, problems = check_gate(rows, baselines)
    if not passed:
        raise SystemExit("Quality gate FAILED:\n" + "\n".join(problems))

    # reconstruct the tokenizer exactly as training built it
    tok = AutoTokenizer.from_pretrained(cfg["pretrained"])
    target_token = target_token_for(cfg, args.init, tok.get_vocab())
    if args.init != "existing":
        tok.add_special_tokens({"additional_special_tokens": [target_token]})
    tgt_id = tok.convert_tokens_to_ids(target_token)

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    embed = model.get_input_embeddings().weight.shape[0]
    if tgt_id >= embed:
        raise SystemExit(
            f"Tokenizer reconstruction mismatch: target id {tgt_id} outside "
            f"the checkpoint embedding ({embed} rows)."
        )
    model.generation_config.forced_bos_token_id = tgt_id

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    model.save_pretrained(staging)          # weights + config, no optimizer state
    tok.save_pretrained(staging)
    (staging / "experiment").mkdir()
    shutil.copy(args.config, staging / "experiment" / "config.yaml")
    rows.to_csv(staging / "experiment" / "scores.csv", index=False)

    best_val = args.best_val or "(see results dataset)"
    card = build_model_card(
        repo_id=repo_id, cfg=cfg, init=args.init, target_token=target_token,
        rows=rows, baselines=baselines, licences=licences,
        model_licence=model_licence, best_val=best_val, commit=git_commit(),
    )
    (staging / "README.md").write_text(card, encoding="utf-8")

    return {"repo_id": repo_id, "staging": staging, "model_licence": model_licence,
            "target_token": target_token, "tgt_id": tgt_id,
            "rows": rows, "baselines": baselines}


def push(staging: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="model")


def run(args) -> None:
    info = assemble(args)
    print(f"\nReady to publish -> {info['repo_id']}")
    print(f"  visibility   : {'private' if args.private else 'public'}")
    print(f"  licence      : {info['model_licence']}")
    print(f"  target token : {info['target_token']} (id {info['tgt_id']})")
    print(f"  staging      : {info['staging']}")
    for _, r in info["rows"].iterrows():
        floor = max(info["baselines"].get(r["book"], {}).values(), default=0.0)
        print(f"    {r['book']}: model {r['best_chrF3']} vs source-copy floor {round(floor, 2)}")
    if args.dry_run:
        print("\nDry run: staging built, nothing pushed.")
        return
    if not args.yes:
        reply = input("\nPush to the Hub? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return
    push(info["staging"], info["repo_id"], private=args.private)
    print(f"Published: https://huggingface.co/{info['repo_id']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Publish an NLLB m2o fine-tune to the Hub")
    p.add_argument("--config", required=True, help="the run's m2o YAML config")
    p.add_argument("--init", required=True, choices=["relative", "scratch", "same_script", "existing"])
    p.add_argument("--checkpoint", required=True, help="checkpoint dir (model.safetensors + config)")
    p.add_argument("--results", default=str(repo_root() / "experiments" / "m2o-matrix-results.csv"))
    p.add_argument("--best-val", default=None, help="best validation chrF3, for the card")
    p.add_argument("--repo", default=None, help="override the repo id")
    p.add_argument("--staging-dir", default=None)
    p.add_argument("--private", action="store_true")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--dry-run", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
