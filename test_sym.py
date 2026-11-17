import sys, time; sys.path.insert(0,'.')
import torch; torch.set_num_threads(8)
from svcore.validate import Validator, fmt
V = Validator()
print("=== symmetric (bidirectional) scoring ===")
print(f"  {'forward only (current)':30s} {fmt(V.run(n_boot=200))}", flush=True)
for w in [0.7, 0.5, 0.3, 0.0]:
    t0=time.time(); r = V.run(n_boot=200, sym=w)
    print(f"  {'w_fwd='+str(w):30s} {fmt(r)}   [{time.time()-t0:.0f}s]", flush=True)
