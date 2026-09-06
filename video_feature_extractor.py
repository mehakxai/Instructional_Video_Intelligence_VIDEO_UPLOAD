
from pathlib import Path
import os
import shutil
import tempfile, subprocess
import wave
import numpy as np


def find_ffmpeg_executable():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg and Path(ffmpeg).exists():
            return str(ffmpeg)
    except Exception:
        pass

    if os.name == "nt":
        candidates = []
        roots = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramW6432", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.path.join(os.environ.get("LOCALAPPDATA", r"C:\Users\Default\AppData\Local"), "Microsoft", "WinGet", "Packages"),
        ]
        for root in roots:
            if not root:
                continue
            base_root = Path(root)
            if base_root.exists():
                candidates.extend(str(p) for p in base_root.rglob("ffmpeg.exe"))
                for pattern in ["Gyan.FFmpeg*", "Gyan.FFmpeg.*", "*FFmpeg*"]:
                    for match in base_root.glob(pattern):
                        candidates.extend(str(p) for p in match.rglob("ffmpeg.exe"))
                for extra in ["Gyan.FFmpeg", "Gyan.FFmpeg.Shared", "Gyan.FFmpeg.Essential"]:
                    p = base_root / extra
                    if p.exists():
                        candidates.extend(str(child) for child in p.rglob("ffmpeg.exe"))
        for candidate in candidates:
            p = Path(candidate)
            if p.exists():
                return str(p)

    return None


def _add_ffmpeg_to_path(ffmpeg):
    ffmpeg_dir = str(Path(ffmpeg).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if ffmpeg_dir not in path_entries:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

def meta(path):
    import cv2
    cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS) or 0; n=cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur=n/fps if fps>0 else 0; cap.release(); return fps,n,dur

def sample_frames(path,every_sec=2.0):
    import cv2
    cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS) or 25
    step=max(int(round(fps*every_sec)),1); out=[]; i=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i%step==0: out.append(frame)
        i+=1
    cap.release(); return out

def extract_visual(path):
    import cv2
    fps,n,dur=meta(path); frames=sample_frames(path)
    if not frames: raise ValueError("No readable video frames.")
    sat=[]; bri=[]; bstd=[]; clarity=[]; warm=[]; motion=[]; flow=[]
    mog=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=16,detectShadows=False)
    cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    prev=None; hits=0
    for frame in frames:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); h,s,v=cv2.split(hsv)
        sat.append(np.mean(s)/255.0); bri.append(np.mean(v)/255.0); bstd.append(np.std(v/255.0))
        clarity.append(np.mean(v>178.5))
        m1=cv2.inRange(hsv,np.array([0,40,40]),np.array([15,255,255]))>0
        m2=cv2.inRange(hsv,np.array([16,40,40]),np.array([40,255,255]))>0
        m3=cv2.inRange(hsv,np.array([170,40,40]),np.array([179,255,255]))>0
        warm.append(np.mean(m1|m2|m3))
        fg=mog.apply(frame); motion.append(np.mean(fg>0))
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl=cv2.calcOpticalFlowFarneback(prev,gray,None,.5,3,15,3,5,1.2,0)
            mag,_=cv2.cartToPolar(fl[...,0],fl[...,1]); flow.append(np.mean(mag))
        prev=gray
        if len(cascade.detectMultiScale(gray,1.1,5,minSize=(40,40)))>0: hits+=1
    return {
        "video_length_min":dur/60,
        "motion_avg":float(np.mean(motion)),
        "saturation_avg":float(np.mean(sat)),
        "brightness_avg":float(np.mean(bri)),
        "brightness_std":float(np.mean(bstd)),
        "clarity_prop":float(np.mean(clarity)),
        "warm_prop":float(np.mean(warm)),
        "flow_mag_avg":float(np.mean(flow)) if flow else 0.0,
        "instructor_face_visible":int(hits>0),
        "face_detected_frame_prop":hits/len(frames)
    }

def scene_length(path):
    _,_,dur=meta(path)
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        video=open_video(str(path)); sm=SceneManager(); sm.add_detector(ContentDetector()); sm.detect_scenes(video)
        n=max(len(sm.get_scene_list()),1)
    except Exception:
        n=1
    return dur/60/n if dur>0 else 0.0

def audio_to_wav(video,wav):
    ffmpeg = find_ffmpeg_executable()
    if ffmpeg is None:
        raise FileNotFoundError(
            "FFmpeg was not found on this system. Install FFmpeg and ensure 'ffmpeg' is on PATH, "
            "or add the FFmpeg bin folder to your system PATH."
        )
    ffmpeg_dir = str(Path(ffmpeg).parent)
    env = os.environ.copy()
    env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")

    p = subprocess.run([ffmpeg, "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if p.returncode == 0:
        return

    err = p.stderr.decode("utf-8", errors="replace").strip() or "unknown error"
    if "Output file does not contain any stream" in err or "matches no streams" in err or "does not contain any stream" in err:
        silent = subprocess.run(
            [ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", "-ar", "16000", "-ac", "1", str(wav)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        if silent.returncode != 0:
            raise RuntimeError(f"FFmpeg silent-audio fallback failed: {silent.stderr.decode('utf-8', errors='replace').strip() or 'unknown error'}")
        return

    raise RuntimeError(f"FFmpeg audio extraction failed: {err}")


def _read_wav_samples(wav):
    with wave.open(str(wav), "rb") as audio:
        frames = audio.readframes(audio.getnframes())
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples

def transcript_features(path,model_name="base"):
    _,_,dur=meta(path)
    try:
        import whisper
    except ImportError:
        return 0.0, 0.0, ""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    with tempfile.TemporaryDirectory() as td:
        wav=Path(td)/"audio.wav"; audio_to_wav(path,wav)
        audio=_read_wav_samples(wav)
        result=whisper.load_model(model_name).transcribe(audio,fp16=False,verbose=False)
    text=result.get("text","") or ""; segs=result.get("segments",[]) or []
    wpm=len(text.split())/(dur/60) if dur>0 else 0.0
    vader=SentimentIntensityAnalyzer()
    scores=[vader.polarity_scores(s.get("text",""))["compound"] for s in segs if s.get("text","").strip()]
    return float(wpm), float(np.mean(scores)) if scores else 0.0, text

def extract_all(path,model_name="base"):
    out=extract_visual(path); out["avg_scene_length_min"]=scene_length(path)
    wpm,sent,text=transcript_features(path,model_name)
    out["speaking_rate_wpm"]=wpm; out["sentiment_avg"]=sent
    return out,text
