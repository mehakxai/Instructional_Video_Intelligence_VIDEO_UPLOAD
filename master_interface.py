
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
REF=json.loads((ROOT/"data"/"reference_results.json").read_text())
FEATURES=REF["features"]; PROFILES=REF["profiles"]
def pos(z):
    return "Relatively low" if z<=-1 else "Below benchmark centre" if z<-.35 else "Near benchmark centre" if z<=.35 else "Above benchmark centre" if z<1 else "Relatively high"
def analyze(vals,face):
    rows=[]
    for f in FEATURES:
        v=float(vals[f["key"]]); z=(v-f["overall_mean"])/f["overall_sd"]
        rows.append({"Characteristic":f["label"],"Value":v,"Unit":f["unit"],"Benchmark mean":f["overall_mean"],"z":z,"Position":pos(z)})
    ds=[]
    for p in PROFILES:
        d=[]
        for f in FEATURES:
            d.append(min(abs(vals[f["key"]]-p[f["key"]])/max(4*f["overall_sd"],1e-12),1))
        d.append(abs(float(face)-p["face_visibility"]))
        ds.append((sum(d)/len(d),p))
    ds.sort(key=lambda x:x[0])
    return rows,ds
