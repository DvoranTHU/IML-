"""
Filename labeling conventions
The Actor id is a 4 digit number at the start of the file.

Actors spoke from a selection of 12 sentences (in parentheses is the three letter acronym used in the second part of the filename):
It's eleven o'clock (IEO).
That is exactly what happened (TIE).
I'm on my way to the meeting (IOM).
I wonder what this is about (IWW).
The airplane is almost full (TAI).
Maybe tomorrow it will be cold (MTI).
I would like a new alarm clock (IWL)
I think I have a doctor's appointment (ITH).
Don't forget a jacket (DFA).
I think I've seen this before (ITS).
The surface is slick (TSI).
We'll stop in a couple of minutes (WSI).

The sentences were presented using different emotion (in parentheses is the three letter code used in the third part of the filename):
Anger (ANG)
Disgust (DIS)
Fear (FEA)
Happy/Joy (HAP)
Neutral (NEU)
Sad (SAD)

and emotion level (in parentheses is the two letter code used in the fourth part of the filename):
Low (LO)
Medium (MD)
High (HI)
Unspecified (XX)
"""

from pathlib import Path
import pandas as pd

EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

LEVEL_MAP = {
    "LO": "low",
    "MD": "medium",
    "HI": "high",
    "XX": "unspecified",
}

# 解析文件名，格式参照以上规则
def parse_filename(stem):
    parts = stem.split("_")
    if len(parts) != 4:
        return None
    actor, sentence, emotion, level = parts
    if emotion not in EMOTION_MAP or level not in LEVEL_MAP:
        return None
    return {
        "actor_id": int(actor),
        "sentence": sentence,
        "emotion_code": emotion,
        "emotion": EMOTION_MAP[emotion],
        "level_code": level,
        "level": LEVEL_MAP[level],
    }

# 构建元数据
def build_metadata(data_dir, project_root):
    rows: list[dict] = []
    for wav in sorted(data_dir.glob("*.wav")):
        parsed = parse_filename(wav.stem)
        if parsed is None:
            continue
        rel = wav.relative_to(project_root).as_posix()
        rows.append(
            {
                "path": rel,
                "dataset": "crema_d",
                "actor_id": parsed["actor_id"],
                "emotion": parsed["emotion"],
                "level": parsed["level"],
                "sentence": parsed["sentence"],
            }
        )
    return pd.DataFrame(rows)
