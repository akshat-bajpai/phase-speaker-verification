import sys, time; sys.path.insert(0,'.')
import torch; torch.set_num_threads(8)
from svcore.validate import Validator, fmt
from svcore import system
from svcore.config import EnrolConfig, GMMConfig, ScoreConfig
V = Validator()
B = system.FINAL_GMM; E = system.FINAL_ENROL
def show(lab, **kw):
    r = V.run(n_boot=200, **kw)
    print(f"  {lab:34s} {fmt(r)}", flush=True); return r
print("=== BASELINE ==="); base = show("current (K=128, r=16, raw)")
print("\n=== relevance factor r ===")
for r_ in [4,8,16,32,64,128,256]:
    show(f"r={r_}", ecfg=EnrolConfig(relevance_factor=float(r_), n_enrol=3))
print("\n=== mixtures K (r=16) ===")
for K in [32,64,128,256,512]:
    show(f"K={K}", gcfg=B.replace(n_components=K))
