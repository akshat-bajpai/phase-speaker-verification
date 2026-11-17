import sys, time; sys.path.insert(0,'.')
import torch; torch.set_num_threads(8)
from svcore.validate import Validator
from svcore import system2
from svcore.config import ScoreConfig
V = Validator(); M = system2.MEMBERS; A = M[0]
P = ("dev","bkg","loo")
def f3(r): return "  ".join(f"{k}={r[k]['eer']*100:5.2f}%" for k in P) + f"   MEAN={r['mean_eer']*100:5.2f}%"
print(f"cohort: {len(V.cohort)} speakers, {len(V.cohort_utterances())} impostor utterances\n", flush=True)
print("=== single system A: normalisation variants ===", flush=True)
t0=time.time(); print(f"  {'raw':26s} {f3(V.run(fcfg=A, protocols=P, n_boot=200))}  [{time.time()-t0:.0f}s]", flush=True)
t0=time.time(); print(f"  {'T-norm (all 150)':26s} {f3(V.run(fcfg=A, scfg=ScoreConfig(norm='tnorm'), protocols=P, n_boot=200))}  [{time.time()-t0:.0f}s]", flush=True)
for top in [20, 50, 100, 150]:
    t0=time.time()
    r = V.run(fcfg=A, protocols=P, n_boot=200, asnorm_top=top, asnorm_sym=True)
    print(f"  {'as-norm sym top-'+str(top):26s} {f3(r)}  [{time.time()-t0:.0f}s]", flush=True)
t0=time.time()
r = V.run(fcfg=A, protocols=P, n_boot=200, asnorm_top=50, asnorm_sym=False)
print(f"  {'as-norm utt-only top-50':26s} {f3(r)}  [{time.time()-t0:.0f}s]", flush=True)
