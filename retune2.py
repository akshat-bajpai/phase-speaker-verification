import sys; sys.path.insert(0,'.')
import torch; torch.set_num_threads(8)
from svcore.validate import Validator, fmt
from svcore import system
from svcore.config import GMMConfig
V = Validator(); F = system.FINAL_FEATURE
def show(lab, fc=None, **kw):
    r = V.run(fcfg=fc, n_boot=200, **kw)
    d = next(iter(V.features(fc or F).values())).shape[1]
    print(f"  {lab:32s} dim={d:3d}  {fmt(r)}", flush=True); return r
print("=== baseline ==="); show("current")
print("\n=== feature dimension / compression ===")
for lab, fc in [("nceps=13 (DCT)", F.replace(n_ceps=13)), ("nceps=20 (DCT)", F.replace(n_ceps=20)),
                ("nbands=24 raw", F.replace(n_bands=24)), ("nbands=40 raw", F.replace(n_bands=40)),
                ("nbands=48 raw", F.replace(n_bands=48)), ("nbands=20 raw", F.replace(n_bands=20)),
                ("no deltas", F.replace(deltas=0)), ("deltas=2", F.replace(deltas=2))]:
    show(lab, fc)
print("\n=== framing ===")
for lab, fc in [("hop2", F.replace(hop_ms=2.0)), ("hop4", F.replace(hop_ms=4.0)),
                ("hop5", F.replace(hop_ms=5.0)), ("win20", F.replace(win_ms=20.0)),
                ("win32", F.replace(win_ms=32.0)), ("nfft1024", F.replace(n_fft=1024))]:
    show(lab, fc)
print("\n=== normalisation / vad ===")
for lab, fc in [("cmvn none", F.replace(cmvn='none')), ("cmvn mv", F.replace(cmvn='mv')),
                ("vad25", F.replace(vad_rel_db=25.)), ("vad40", F.replace(vad_rel_db=40.)),
                ("magw off", F.replace(mag_weight=False))]:
    show(lab, fc)
