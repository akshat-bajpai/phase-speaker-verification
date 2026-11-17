import sys, time; sys.path.insert(0,'.')
import torch; torch.set_num_threads(8)
from svcore.validate import Validator, fmt
from svcore import system
V = Validator(); F = system.FINAL_FEATURE
A = F.replace(n_fft=1024, win_ms=16.0, vad_rel_db=25.0)   # best single
B = F.replace(n_fft=2048, win_ms=25.0, vad_rel_db=25.0)
C = F.replace(n_fft=1024, win_ms=20.0, vad_rel_db=25.0)
D = F.replace(n_fft=1024, win_ms=25.0, vad_rel_db=30.0)
GD = F.replace(stream='mgdcc', n_bands=48, n_ceps=30, mgd_denom='exact', n_fft=1024)
FD = F.replace(stream='fdpd', n_fft=1024, win_ms=20.0, vad_rel_db=25.0)
print("=== singles ===")
for lab, fc in [("A nfft1024 win16", A), ("B nfft2048 win25", B), ("C nfft1024 win20", C),
                ("D nfft1024 win25", D), ("GD exact nfft1024", GD), ("FDPD nfft1024 win20", FD)]:
    print(f"  {lab:24s} {fmt(V.run(fcfg=fc, n_boot=200))}", flush=True)
print("\n=== fusion (equal weights, z-scored) ===")
for lab, group in [("A+B", [A,B]), ("A+C", [A,C]), ("A+B+C", [A,B,C]), ("A+B+C+D", [A,B,C,D]),
                   ("A+B+GD", [A,B,GD]), ("A+B+C+GD", [A,B,C,GD]),
                   ("A+B+C+FD", [A,B,C,FD]), ("A+B+C+GD+FD", [A,B,C,GD,FD])]:
    t0=time.time(); r = V.run_fusion(group, n_boot=200)
    print(f"  {lab:16s} {fmt(r)}   [{time.time()-t0:.0f}s]", flush=True)
