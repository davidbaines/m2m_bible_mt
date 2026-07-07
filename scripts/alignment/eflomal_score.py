import sys
from eflomal import Aligner
src, trg = sys.argv[1], sys.argv[2]
fwd, rev = src + ".sf", src + ".sr"
a = Aligner()
with open(src, encoding="utf-8") as sf, open(trg, encoding="utf-8") as tf:
    a.align(sf, tf, scores_filename_fwd=fwd, scores_filename_rev=rev, quiet=True)
def per_token(scores_file, text_file):
    sc = [float(x) for x in open(scores_file, encoding="utf-8").read().split("\n") if x.strip()]
    toks = [len(l.split()) for l in open(text_file, encoding="utf-8").read().split("\n") if l.strip()]
    n = min(len(sc), len(toks)); pairs = [(sc[i], toks[i]) for i in range(n) if toks[i] > 0]
    return sum(s for s, _ in pairs) / sum(t for _, t in pairs)
print(round((per_token(fwd, trg) + per_token(rev, src)) / 2, 4))
