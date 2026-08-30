#!/usr/bin/env python3
"""Task 7 recalibration: is R still right once direct output removes the
per-tile staging cost?

Sweeps every tile allocation exhaustively against auto at R=10 and R=25, over
thread counts and prompt sizes not used to pick either value, all with direct
output ON. Produces artifacts/direct-output/cost_model_recal.csv.
"""
import csv, os, re, statistics as st, subprocess, sys
from pathlib import Path
BIN="refs/BitNet/build-xdna3/bin/llama-bench"; MODEL="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART=os.path.abspath("artifacts/xclbin-tuned"); DPT=323.4
def run(prompt, th, tiles, R, inner=2):
    env=dict(os.environ, BITNET_XDNA="1", BITNET_XDNA_ARTIFACTS=ART, BITNET_XDNA_STATS="1",
             BITNET_XDNA_DIRECT_OUT="1", BITNET_XDNA_ASYNC="0")
    if tiles is not None: env["BITNET_XDNA_TILES"]=str(tiles)
    if R is not None: env["BITNET_XDNA_NPU_THREADS"]=str(R)
    p=subprocess.run([BIN,"-m",MODEL,"-p",str(prompt),"-n","0","-t",str(th),"-ngl","0",
                      "-r",str(inner),"-ub",str(prompt)],capture_output=True,text=True,env=env,timeout=3600)
    o=p.stdout+p.stderr
    t=re.search(rf"pp{prompt} \|\s*([0-9.]+)",o); d=re.search(r"dispatches=(\d+)",o)
    return (float(t.group(1)) if t else None, round(int(d.group(1))/(inner+1)/DPT) if d else 0)
rows=[]; acc={}
cells=[(2048,th) for th in (4,6,8,10,12,15)]+[(3072,th) for th in (4,8,15)]
for rep in (1,2):
    for prompt,th in cells:
        mt=prompt//1024
        for t in list(range(mt+1)):
            tok,_=run(prompt,th,t,None); acc.setdefault((prompt,th,t),[]).append(tok)
            rows.append(dict(rep=rep,prompt=prompt,threads=th,mode=f"tiles{t}",R="",tok_s=tok,pick=t))
        for R in (10,25):
            tok,pick=run(prompt,th,None,R); acc.setdefault((prompt,th,f"R{R}"),[]).append(tok)
            acc.setdefault((prompt,th,f"P{R}"),[]).append(pick)
            rows.append(dict(rep=rep,prompt=prompt,threads=th,mode="auto",R=R,tok_s=tok,pick=pick))
        print(f"  [{rep}] pp{prompt} t{th} done",flush=True)
with open("artifacts/direct-output/cost_model_recal.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\n{'prompt':>7}{'th':>4}{'best':>7}{'bestT':>7}{'R=10':>9}{'pick':>5}{'reg':>7}{'R=25':>9}{'pick':>5}{'reg':>7}")
r10=[]; r25=[]
for prompt,th in cells:
    mt=prompt//1024
    fx={t:st.median(acc[(prompt,th,t)]) for t in range(mt+1)}
    bt=max(fx,key=fx.get); best=fx[bt]
    a10=st.median(acc[(prompt,th,"R10")]); p10=round(st.median(acc[(prompt,th,"P10")]))
    a25=st.median(acc[(prompt,th,"R25")]); p25=round(st.median(acc[(prompt,th,"P25")]))
    r10.append(best/a10); r25.append(best/a25)
    print(f"{prompt:>7}{th:>4}{best:>7.1f}{bt:>7}{a10:>9.1f}{p10:>5}{best/a10:>6.3f}x{a25:>9.1f}{p25:>5}{best/a25:>6.3f}x")
print(f"\n  mean regret  R=10 {st.mean(r10):.3f}x   R=25 {st.mean(r25):.3f}x")
print(f"  worst regret R=10 {max(r10):.3f}x   R=25 {max(r25):.3f}x")
