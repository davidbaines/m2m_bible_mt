import csv, itertools, subprocess, tempfile, yaml
from pathlib import Path
from samileides.data import load_verses, repo_root
from samileides.align_score import parallel_tokens

SP = "/tmp/claude-1000/-home-david-Documents-Github/95a24bdc-9442-45d9-9e43-b9fb7fe88cf1/scratchpad"
PY = f"{SP}/eflomal-venv/bin/python"
HELPER = f"{SP}/eflomal_score.py"


def write_lines(sents, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(" ".join(s) for s in sents) + "\n")


def score_pair(a_sents, b_sents):
    with tempfile.TemporaryDirectory() as d:
        write_lines(a_sents, f"{d}/a.txt"); write_lines(b_sents, f"{d}/b.txt")
        out = subprocess.run([PY, HELPER, f"{d}/a.txt", f"{d}/b.txt"],
                             capture_output=True, text=True)
        return float(out.stdout.strip().split("\n")[-1])


rows = []
for name in ["ton", "nde", "mdy", "rmc", "control_ilo"]:
    cfg = yaml.safe_load((repo_root() / "configs" / "experiments" / "m2o" / f"{name}.yaml").read_text())
    tgt = cfg["target"]; srcs = cfg["sources"]
    verses = load_verses([tgt["tid"]] + [s["tid"] for s in srcs])
    for s in srcs:
        a, b = parallel_tokens(verses, s["tid"], tgt["tid"])
        sc = score_pair(a, b)
        rows.append({"experiment": name, "pair": f"{s['code']}->{tgt['code']}", "kind": "source-target",
                     "same_script": s["flores"].split("_")[1] == tgt["script"], "n": len(a), "eflomal_score": sc})
        print(f"{name} {s['code']}->{tgt['code']} eflomal={sc}", flush=True)
    for s1, s2 in itertools.combinations(srcs, 2):
        a, b = parallel_tokens(verses, s1["tid"], s2["tid"])
        sc = score_pair(a, b)
        rows.append({"experiment": name, "pair": f"{s1['code']}~{s2['code']}", "kind": "source-source",
                     "same_script": s1["flores"].split("_")[1] == s2["flores"].split("_")[1], "n": len(a), "eflomal_score": sc})
        print(f"{name} {s1['code']}~{s2['code']} eflomal={sc}", flush=True)

out = repo_root() / "experiments" / "m2o-alignment-eflomal.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE", out)

# compare closest source: IBM-1 (higher=better) vs eflomal (lower=better)
import pandas as pd
ib = pd.read_csv(repo_root() / "experiments" / "m2o-alignment.csv")
ef = pd.DataFrame(rows)
print("\n=== closest source to each target ===")
for name in ["ton", "nde", "mdy", "rmc", "control_ilo"]:
    ibs = ib[(ib.experiment == name) & (ib.kind == "source-target")]
    efs = ef[(ef.experiment == name) & (ef.kind == "source-target")]
    ib_best = ibs.loc[ibs.align_score.idxmax(), "pair"] if len(ibs) else "?"
    ef_best = efs.loc[efs.eflomal_score.idxmin(), "pair"]
    print(f"{name}: IBM-1 -> {ib_best} | eflomal -> {ef_best} | {'SAME' if ib_best==ef_best else 'DIFFERENT'}")
