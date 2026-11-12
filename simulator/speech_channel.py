"""Speech channel for text runs: the agent hears the caller only through TTS -> phone -> ASR.

With `run_text.py --speech-channel phone_noisy`, every caller turn is
    spoken by Amazon Polly (bilingual Indian voice: en-IN for English callers, hi-IN for
    Hinglish callers; one voice per call), passed through the simulated phone channel
    (8 kHz, 300-3400 Hz, mu-law; optional pink noise), transcribed by Amazon Transcribe
    (automatic language identification over en-IN + hi-IN),
and the agent receives only the transcript -- what a cascade voice agent (STT -> LLM -> TTS)
actually gets. Control tokens (###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###) are kept
outside the audio and re-attached after transcription. The caller's original words stay in
the run's speech log for analysis.

The agent's replies stay text (the caller is an LLM reading them), so this isolates the
hearing side of a voice call. Costs: Polly characters and Transcribe seconds are metered and
added to the run's spend guard.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.request
import uuid
import wave
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np

from simulator.hearing import REGION, TAG, phone_channel

TOKEN = re.compile(r"###(?:STOP|TRANSFER|OUT-OF-SCOPE)###")
VOICES = (("Kajal", "neural"), ("Arjun", "generative"))
POLLY_USD_PER_CHAR = {"neural": 16e-6, "generative": 30e-6}
TRANSCRIBE_USD_PER_SEC = 0.0001  # standard batch, us-east-1, Nov 2026
LANG_OPTIONS = ["en-IN", "hi-IN"]
SR = 16000


@dataclass
class SpeechMeter:
    usd: float = 0.0
    turns: int = 0
    chars: int = 0
    audio_seconds: float = 0.0
    failures: int = 0
    log: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, *, chars: int, engine: str, seconds: float, entry: dict) -> None:
        with self.lock:
            self.turns += 1
            self.chars += chars
            self.audio_seconds += seconds
            self.usd += chars * POLLY_USD_PER_CHAR[engine] + seconds * TRANSCRIBE_USD_PER_SEC
            self.log.append(entry)

    @property
    def total(self) -> float:
        return self.usd

    def summary(self) -> dict:
        return {
            "speech_turns": self.turns,
            "speech_chars": self.chars,
            "speech_audio_seconds": round(self.audio_seconds, 1),
            "speech_usd": round(self.usd, 4),
            "speech_failures": self.failures,
        }


class SpeechChannel:
    def __init__(self, condition: str, snr_db: float, bucket: str, meter: SpeechMeter):
        import boto3

        self.condition = condition
        self.snr_db = snr_db
        self.bucket = bucket
        self.meter = meter
        self._local = threading.local()
        self._boto3 = boto3

    def _clients(self):
        if not hasattr(self._local, "c"):
            s = self._boto3.Session(region_name=REGION)
            self._local.c = (s.client("polly"), s.client("s3"), s.client("transcribe"))
        return self._local.c

    def hear(self, text: str, hinglish: bool, call_key: str) -> str:
        """The caller's turn as the agent would hear it. Tokens are kept, speech is transcribed."""
        tokens = TOKEN.findall(text or "")
        spoken = TOKEN.sub(" ", text or "").strip()
        if not spoken:
            return text
        h = int(hashlib.sha1(call_key.encode()).hexdigest(), 16)
        voice, engine = VOICES[h % len(VOICES)]
        lang = "hi-IN" if hinglish else "en-IN"
        polly, s3, tr = self._clients()
        try:
            audio = polly.synthesize_speech(
                Text=spoken[:2900], OutputFormat="pcm", SampleRate=str(SR),
                VoiceId=voice, Engine=engine, LanguageCode=lang,
            )["AudioStream"].read()
            x = np.frombuffer(audio, dtype=np.int16).astype(np.float64)
            noisy = self.condition == "phone_noisy"
            y = phone_channel(x, noisy=noisy, seed=h % (2**31), snr_db=self.snr_db)
            buf = BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(8000)
                w.writeframes(np.clip(y, -32768, 32767).astype(np.int16).tobytes())
            key = f"speech/{uuid.uuid4().hex}.wav"
            s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())
            job = f"ccb-sc-{uuid.uuid4().hex}"
            tr.start_transcription_job(
                TranscriptionJobName=job, Media={"MediaFileUri": f"s3://{self.bucket}/{key}"},
                MediaFormat="wav", IdentifyMultipleLanguages=True, LanguageOptions=LANG_OPTIONS,
                Tags=TAG,
            )
            heard = None
            for _ in range(240):  # up to ~8 minutes
                st = tr.get_transcription_job(TranscriptionJobName=job)["TranscriptionJob"]
                if st["TranscriptionJobStatus"] == "COMPLETED":
                    data = json.loads(urllib.request.urlopen(st["Transcript"]["TranscriptFileUri"]).read())
                    heard = data["results"]["transcripts"][0]["transcript"]
                    break
                if st["TranscriptionJobStatus"] == "FAILED":
                    raise RuntimeError(st.get("FailureReason"))
                time.sleep(2)
            if heard is None:
                raise TimeoutError("transcription did not finish")
            try:
                s3.delete_object(Bucket=self.bucket, Key=key)
            except Exception:  # noqa: BLE001 -- cleanup only
                pass
        except Exception as e:  # noqa: BLE001 -- a failed turn passes through, and is counted
            with self.meter.lock:
                self.meter.failures += 1
            print(f"  speech channel failed ({type(e).__name__}: {str(e)[:100]}); passing text through")
            return text
        seconds = len(y) / 8000
        self.meter.add(
            chars=len(spoken), engine=engine, seconds=max(seconds, 15.0),  # Transcribe bills >= 15 s
            entry={"call": call_key, "voice": f"{voice}/{engine}/{lang}", "said": spoken,
                   "heard": heard, "tokens": tokens},
        )
        return (heard.strip() + (" " + " ".join(tokens) if tokens else "")).strip() or text


def bucket_for(session) -> str:
    from simulator.hearing import ensure_bucket

    return ensure_bucket(session)
