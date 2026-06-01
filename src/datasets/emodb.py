"""
Every utterance is named according to the same scheme:
Positions 1-2: number of speaker
Positions 3-5: code for text
Position 6: emotion (sorry, letter stands for german emotion word)
Position 7: if there are more than two versions these are numbered a, b, c ....
Example: 03a01Fa.wav is the audio file from Speaker 03 speaking text a01 with the emotion "Freude" (Happiness).

letter	emotion (english)	letter	emotion (german)
A	anger	W	Ärger (Wut)
B	boredom	L	Langeweile
D	disgust	E	Ekel
F	anxiety/fear	A	Angst
H	happiness	F	Freude
S	sadness	T	Trauer
N = neutral version
"""

import pandas as pd

EMOTION_MAP = {
    "N": "neutral",
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
}

# 解析文件名，格式如上
def parse_filename(stem):
    if len(stem) < 6:
        return None
    speaker = stem[0:2]
    if not speaker.isdigit():
        return None
    emotion_code = stem[5]
    if emotion_code not in EMOTION_MAP:
        return None
    return {
        "actor_id": int(speaker),
        "session": stem[2],
        "emotion_code": emotion_code,
        "emotion": EMOTION_MAP[emotion_code],
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
                "dataset": "emodb",
                "actor_id": parsed["actor_id"],
                "emotion": parsed["emotion"],
            }
        )
    return pd.DataFrame(rows)
