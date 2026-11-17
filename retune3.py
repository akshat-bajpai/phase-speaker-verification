import sys; sys.path.insert(0,'.')
import torch; torch.set_num_threads(8)
from svcore.validate import Validator, fmt
from svcore import system
V = Validator(); F = system.FINAL_FEATURE
def show(lab, fc):
    r = V.run(fcfg=fc, n_boot=200)
    d = next(iter(V.features(fc).values())).shape[1]
    print(f"  {lab:38s} dim={d:3d}  {fmt(r)}", flush=True); return r['mean_eer'], fc
best = (1.0, None); res=[]
print("=== combining nfft / win / vad ===")
for nfft in [1024, 2048]:
    for win in [16.0, 20.0, 25.0]:
        for vad in [25.0, 30.0]:
            fc = F.replace(n_fft=nfft, win_ms=win, vad_rel_db=vad)
            m, _ = show(f"nfft{nfft} win{win:g} vad{vad:g}", fc)
            res.append((m, f"nfft{nfft} win{win:g} vad{vad:g}", fc))
res.sort(key=lambda x: x[0])
print("\n  TOP 3:", [f"{r[1]} ({r[0]*100:.2f}%)" for r in res[:3]])
print("\n=== hop, around the best setting ===")
_, _, bfc = res[0]
for hop in [3.0, 4.0, 5.0, 6.0]:
    show(f"hop{hop:g} + best", bfc.replace(hop_ms=hop))
