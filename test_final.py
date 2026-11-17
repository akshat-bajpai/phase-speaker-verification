import sys, time; sys.path.insert(0,'.')
import torch; torch.set_num_threads(8)
from svcore.validate import Validator
from svcore import system2
from svcore.config import ScoreConfig
V = Validator()
M = system2.MEMBERS
P = ("dev","bkg")          # both enrolment-size 3, matching the eval condition
def f2(r): return "  ".join(f"{k}={r[k]['eer']*100:5.2f}%" for k in P) + f"   MEAN={r['mean_eer']*100:5.2f}%"
print("=== candidates (enrol-size-3 protocols only) ===", flush=True)
t0=time.time(); print(f"  {'A alone, raw':28s} {f2(V.run(fcfg=M[0], protocols=P, n_boot=200))}  [{time.time()-t0:.0f}s]", flush=True)
t0=time.time(); print(f"  {'A alone, T-norm':28s} {f2(V.run(fcfg=M[0], scfg=ScoreConfig(norm='tnorm'), protocols=P, n_boot=200))}  [{time.time()-t0:.0f}s]", flush=True)
t0=time.time(); print(f"  {'fusion A+B+C+FD, raw':28s} {f2(V.run_fusion(M, protocols=P, n_boot=200))}  [{time.time()-t0:.0f}s]", flush=True)
t0=time.time(); r=V.run_fusion(M, scfg=ScoreConfig(norm='tnorm'), protocols=P, n_boot=200)
print(f"  {'fusion A+B+C+FD, T-norm':28s} {f2(r)}  [{time.time()-t0:.0f}s]", flush=True)
for k in P: print(f"      {k}: EER={r[k]['eer']*100:.2f}% [{r[k]['ci'][0]*100:.2f}-{r[k]['ci'][1]*100:.2f}] AUC={r[k]['auc']:.4f}")
