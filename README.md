# Direct Video Upload Version

This is the version you asked for: a user uploads a raw instructional video and the app automatically extracts the final Objective 3 characteristics.

Workflow:
**Upload video → feature extraction → benchmark positioning → closest empirical design profile → downloadable report**

Local run:
1. Install Python 3.10/3.11.
2. Install FFmpeg.
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

For a seminar, local execution is safer than free cloud hosting because Whisper + PyTorch are memory intensive.

Final features:
video length, average scene length, speaking rate, transcript sentiment, motion intensity, saturation, brightness, brightness variability, clarity proxy, warm-colour proportion, optical-flow magnitude, instructor visibility.

Optical-flow direction is excluded.
