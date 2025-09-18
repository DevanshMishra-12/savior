#ai_voice.py
import os
import shutil
import subprocess
import time
from pathlib import Path
from hashlib import sha1
from typing import Optional, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from gtts import gTTS
from pydub import AudioSegment
from pydub.generators import Sine

FFMPEG_BIN = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_BIN

AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")

AudioPath = Path("tts_cache")
AudioPath.mkdir(exist_ok=True)

Language = {"en": "en", "hi": "hi"}
SpeakCmd = shutil.which("speak")  # fallback if system TTS available

TEMPLATES = {
    "en": {
        "reassure": "Rescue is coming. Move to higher ground and wave at the drone.",
        "eta": "Rescue boat will arrive at {landmark} in {mins} minutes.",
        "hazard": "Strong current ahead. Stay on the roof."
    },
    "hi": {
        "reassure": "मदद आ रही है। ऊँचे स्थान पर जाएँ और ड्रोन की ओर हाथ हिलाएँ।",
        "eta": "राहत नाव {landmark} पर {mins} मिनट में पहुँचेगी।",
        "hazard": "तेज़ धारा आगे है। पानी में मत उतरें। छत पर रहें।"
    },
}

def Feature(text: str, lang: str = "en", rate_wpm: int = 160) -> Path:
    key = sha1(f"{lang}:{rate_wpm}:{text}".encode()).hexdigest()[:16]
    Output = AudioPath / f"{key}.wav"
    if Output.exists():
        return Output

    if SpeakCmd:
        try:
            cmd = [SpeakCmd, "l", Language.get(lang, "en"), "s", str(rate_wpm), "o", str(Output), "T", text]
            subprocess.run(cmd, check=True)
            return Output
        except Exception:
            pass

    if lang in ["en", "hi"]:
        mp3_tmp = AudioPath / f"{key}.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(mp3_tmp)
        AudioSegment.from_mp3(mp3_tmp).export(Output, format="wav")
        mp3_tmp.unlink(missing_ok=True)
        return Output

    raise RuntimeError(f"No TTS available for language '{lang}'")

def Alert(ms=500, freq=1000, gain_db=-3):
    return Sine(freq).to_audio_segment(duration=ms).apply_gain(gain_db)

def Broadcasting(text: str, lang: str = "en") -> Path:
    Text = Feature(text, lang)
    speech = AudioSegment.from_wav(Text)
    beep = Alert(500, 1000)
    Audio = beep + AudioSegment.silent(duration=100) + speech

    TTS = Text.with_name(Text.stem + "_final.wav")
    Audio.export(TTS, format="wav")

    subprocess.run([os.path.join(FFMPEG_BIN, "ffplay.exe"), "-nodisp", "-autoexit", str(TTS)])
    time.sleep(0.2)
    return TTS

def say(msg_key: str, lang: str = "en", **kwargs) -> Path:
    text = TEMPLATES.get(lang, TEMPLATES["en"]).get(msg_key, "")
    text = text.format(**kwargs)
    return Broadcasting(text, lang)

def say_all_languages(msg_key: str, **kwargs):
    for lang in ["en", "hi"]:
        try:
            say(msg_key, lang, **kwargs)
        except RuntimeError:
            continue

app = FastAPI()

class SayReq(BaseModel):
    key: Optional[str] = None
    text: Optional[str] = None
    lang: str = "en"
    params: Dict = {}

@app.post("/say")
def API(req: SayReq):
    if req.text:
        path = Broadcasting(req.text, req.lang)
    else:
        path = say(req.key, req.lang, **req.params)
    return {"file": str(path)}

if __name__ == "__main__":
    say_all_languages("reassure")
    say_all_languages("eta", landmark="You", mins=5)
    say_all_languages("hazard")
    Broadcasting("Stay calm, rescue is on the way", "en")
    Broadcasting("शांत रहें, बचाव कार्य जारी है", "hi")