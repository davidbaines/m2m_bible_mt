import csv, itertools, yaml
from pathlib import Path
from samileides.data import load_verses, repo_root
from samileides.align_score import alignability, parallel_tokens

configs = ["ton", "nde", "mdy", "rmc", "control_ilo"]
rows = []
for name in configs:
    cfg = yaml.safe_load((repo_root() / "configs" / "experiments" / "m2o" / f"{name}.yaml").read_text())
    tgt = cfg["target"]
    srcs = cfg["sources"]
    tids = [tgt["tid"]] + [s["tid"] for s in srcs]
    verses = load_verses(tids)
    tgt_script = tgt["script"]
    # source -> target
    for s in srcs:
        a, b = parallel_tokens(verses, s["tid"], tgt["tid"])
        sc = alignability(a, b)
        same = s["flores"].split("_")[1] == tgt_script
        rows.append({"experiment": name, "pair": f"{s['code']}->{tgt['code']}",
                     "kind": "source-target", "same_script": same, "n": len(a), "align_score": sc})
        print(f"{name} {s['code']}->{tgt['code']} same_script={same} score={sc}", flush=True)
    # source <-> source
    for s1, s2 in itertools.combinations(srcs, 2):
        a, b = parallel_tokens(verses, s1["tid"], s2["tid"])
        sc = alignability(a, b)
        same = s1["flores"].split("_")[1] == s2["flores"].split("_")[1]
        rows.append({"experiment": name, "pair": f"{s1['code']}~{s2['code']}",
                     "kind": "source-source", "same_script": same, "n": len(a), "align_score": sc})
        print(f"{name} {s1['code']}~{s2['code']} same_script={same} score={sc}", flush=True)

out = repo_root() / "experiments" / "m2o-alignment.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("WROTE", out)
