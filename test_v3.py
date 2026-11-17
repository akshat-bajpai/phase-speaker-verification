import sys, time; sys.path.insert(0,'.')
import torch; torch.set_num_threads(8)
from svcore.validate import Validator
from svcore import system2
from svcore.config import ScoreConfig
V = Validator(); M = system2.MEMBERS
P = ("dev","bkg","loo")
def f3(r): return "  ".join(f"{k}={r[k]['eer']*100:5.2f}%" for k in P) + f"   MEAN={r['mean_eer']*100:5.2f}%"
print("=== FUSION: normalisation variants (all 3 protocols) ===", flush=True)
t0=time.time(); print(f"  {'fusion raw':26s} {f3(V.run_fusion(M, protocols=P, n_boot=300))}  [{time.time()-t0:.0f}s]", flush=True)
t0=time.time(); print(f"  {'fusion + T-norm (v2)':26s} {f3(V.run_fusion(M, scfg=ScoreConfig(norm='tnorm'), protocols=P, n_boot=300))}  [{time.time()-t0:.0f}s]", flush=True)
for top in [50, 100, 150]:
    t0=time.time()
    r = V.run_fusion(M, protocols=P, n_boot=300, asnorm_top=top)
    print(f"  {'fusion + as-norm top-'+str(top):26s} {f3(r)}  [{time.time()-t0:.0f}s]", flush=True)
