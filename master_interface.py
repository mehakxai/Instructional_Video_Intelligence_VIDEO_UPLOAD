
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent

# Keep deployment working even when a hosting provider omits the repository data file.
_DEFAULT_REFERENCE_RESULTS={
    "features":[
        {"key":"video_length_min","label":"Video length","unit":"min","overall_mean":5.572,"overall_sd":2.276},
        {"key":"avg_scene_length_min","label":"Average scene length","unit":"min","overall_mean":2.143,"overall_sd":2.345},
        {"key":"speaking_rate_wpm","label":"Speaking rate","unit":"WPM","overall_mean":157.111,"overall_sd":47.185},
        {"key":"sentiment_avg","label":"Transcript sentiment","unit":"VADER compound","overall_mean":0.116,"overall_sd":0.09},
        {"key":"motion_avg","label":"Motion intensity","unit":"proportion","overall_mean":0.188,"overall_sd":0.163},
        {"key":"saturation_avg","label":"Average saturation","unit":"0-1","overall_mean":0.16,"overall_sd":0.133},
        {"key":"brightness_avg","label":"Average brightness","unit":"0-1","overall_mean":0.545,"overall_sd":0.309},
        {"key":"brightness_std","label":"Brightness variability","unit":"0-1","overall_mean":0.192,"overall_sd":0.074},
        {"key":"clarity_prop","label":"Clarity proportion","unit":"brightness-based proxy","overall_mean":0.457,"overall_sd":0.34},
        {"key":"warm_prop","label":"Warm-colour proportion","unit":"proportion","overall_mean":0.073,"overall_sd":0.087},
        {"key":"flow_mag_avg","label":"Optical-flow magnitude","unit":"pixel displacement proxy","overall_mean":2.166,"overall_sd":1.886},
    ],
    "profiles":[
        {"profile":1,"n":48,"name":"Bright clarity-oriented instructor-present","face_visibility":0.75,"video_length_min":5.372979,"avg_scene_length_min":1.883688,"speaking_rate_wpm":147.423,"sentiment_avg":0.10975,"motion_avg":0.134771,"saturation_avg":0.121125,"brightness_avg":0.752063,"brightness_std":0.204021,"clarity_prop":0.708188,"warm_prop":0.035771,"flow_mag_avg":1.637458,"rating_mean":4.5771},
        {"profile":2,"n":33,"name":"Dynamic high-motion instructor-led","face_visibility":1.0,"video_length_min":5.414848,"avg_scene_length_min":0.651242,"speaking_rate_wpm":161.632,"sentiment_avg":0.14703,"motion_avg":0.355697,"saturation_avg":0.266697,"brightness_avg":0.473152,"brightness_std":0.223152,"clarity_prop":0.285121,"warm_prop":0.161515,"flow_mag_avg":3.991818,"rating_mean":4.4606},
        {"profile":3,"n":19,"name":"Low-motion long-scene screen-oriented","face_visibility":0.052632,"video_length_min":6.350316,"avg_scene_length_min":5.388211,"speaking_rate_wpm":173.733,"sentiment_avg":0.079684,"motion_avg":0.033158,"saturation_avg":0.073789,"brightness_avg":0.145579,"brightness_std":0.109737,"clarity_prop":0.120368,"warm_prop":0.011263,"flow_mag_avg":0.331526,"rating_mean":4.5105},
    ],
}


def _load_reference_results():
    candidates=(
        ROOT/"data"/"reference_results.json",
        Path.cwd()/"data"/"reference_results.json",
        ROOT/"reference_results.json",
        Path.cwd()/"reference_results.json",
    )
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return _DEFAULT_REFERENCE_RESULTS


REF=_load_reference_results()
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
