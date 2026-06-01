"""
File naming convention
Each of the 7356 RAVDESS files has a unique filename.
The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics:

Filename identifiers
Modality (01 = full-AV, 02 = video-only, 03 = audio-only). 
Vocal channel (01 = speech, 02 = song). 
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised). 
Emotional intensity (01 = normal, 02 = strong). 
NOTE: There is no strong intensity for the 'neutral' emotion. 
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"). 
Repetition (01 = 1st repetition, 02 = 2nd repetition). 
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Filename example: 02-01-06-01-02-01-12.mp4
Video-only (02) Speech (01) Fearful (06) Normal intensity (01) Statement "dogs" (02) 1st Repetition (01) 12th Actor (12) Female, as the actor ID number is even.
"""

import pandas as pd

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

AUDIO_MODALITY = "03"
SPEECH_CHANNEL = "01"

# 解析文件名，格式如上
def parse_filename(stem):
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if modality != AUDIO_MODALITY or channel != SPEECH_CHANNEL:
        return None
    if emotion not in EMOTION_MAP:
        return None
    return {
        "modality": modality,
        "channel": channel,
        "emotion_code": emotion,
        "emotion": EMOTION_MAP[emotion],
        "intensity_code": intensity,
        "intensity": "normal" if intensity == "01" else "strong",
        "statement_code": statement,
        "statement": "kids" if statement == "01" else "dogs",
        "repetition_code": repetition,
        "repetition": 1 if repetition == "01" else 2,
        "actor_id": int(actor),
    }

# 构建元数据
def build_metadata(data_dir, project_root):
    rows: list[dict] = []
    for wav in sorted(data_dir.rglob("*.wav")):
        parsed = parse_filename(wav.stem)
        if parsed is None:
            continue
        rel = wav.relative_to(project_root).as_posix()
        rows.append(
            {
                "path": rel,
                "dataset": "ravdess",
                "actor_id": parsed["actor_id"],
                "emotion": parsed["emotion"],
                "intensity": parsed["intensity"],
                "statement": parsed["statement"],
                "repetition": parsed["repetition"],
            }
        )
    return pd.DataFrame(rows)
