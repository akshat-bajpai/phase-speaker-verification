import sys, time; sys.path.insert(0,'.')
import torch; torch.set_num_threads(8)
from svcore.validate import Validator, fmt
from svcore import system, data, scoring as SC, protocols as P
from svcore.config import ScoreConfig
V = Validator(); F = system.FINAL_FEATURE
A = F.replace(n_fft=1024, win_ms=16.0, vad_rel_db=25.0)
B = F.replace(n_fft=2048, win_ms=25.0, vad_rel_db=25.0)
C = F.replace(n_fft=1024, win_ms=20.0, vad_rel_db=25.0)
D = F.replace(n_fft=1024, win_ms=25.0, vad_rel_db=30.0)
print("=== T-norm on the honest protocols ===")
print(f"  {'A raw':22s} {fmt(V.run(fcfg=A, n_boot=200))}", flush=True)
t0=time.time()
print(f"  {'A + T-norm':22s} {fmt(V.run(fcfg=A, scfg=ScoreConfig(norm='tnorm'), n_boot=200))}  [{time.time()-t0:.0f}s]", flush=True)
t0=time.time()
print(f"  {'A + Z-norm':22s} {fmt(V.run(fcfg=A, scfg=ScoreConfig(norm='znorm'), n_boot=200))}  [{time.time()-t0:.0f}s]", flush=True)
print("\n=== fusion + T-norm ===")
t0=time.time()
print(f"  {'A+B+C+D raw':22s} {fmt(V.run_fusion([A,B,C,D], n_boot=200))}", flush=True)
t0=time.time()
r = V.run_fusion([A,B,C,D], scfg=ScoreConfig(norm='tnorm'), n_boot=200)
print(f"  {'A+B+C+D + T-norm':22s} {fmt(r)}  [{time.time()-t0:.0f}s]", flush=True)
