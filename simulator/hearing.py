"""Stage B "hearing" test: can speech-to-text catch what callers say on a noisy phone line?

Caller lines that carry the facts a front desk must get right -- name, hospital ID (UHID), date
of birth, mobile number, and the red-flag symptoms -- are spoken by Amazon Polly in English and
Hinglish, passed through a simulated phone channel, transcribed by Amazon Transcribe, and scored.

Steps (each reads/writes runs/hearing/):
    build       items.json from the generated tasks (no AWS)
    synth       Polly speech for every item (16 kHz PCM)            -- AWS, ~$1
    channel     clean / phone / phone_noisy WAVs (no AWS)
    transcribe  Transcribe batch jobs per condition x language config -- AWS, ~$5
    score       entity capture and red-flag keyword recall (no AWS)

    python -m simulator.hearing build|synth|channel|transcribe|score [--limit N]

Phone channel (numpy only, mirrors tau2's telephony settings): resample to 8 kHz, 300-3400 Hz
band-pass, mu-law encode/decode, 0.9 headroom. "phone_noisy" adds pink noise at 10 dB SNR.
All audio and transcripts stay under runs/ (gitignored). AWS: us-east-1, S3 bucket
carecall-bench-hearing-<account>, tagged project=carecall-bench; nothing else is created.
"""

from __future__ import annotations

import argparse
import audioop  # stdlib in 3.12 (deprecated in 3.13); used only for mu-law
import io
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "hearing"
REGION = "us-east-1"
TAG = [{"Key": "project", "Value": "carecall-bench"}]
CONDITIONS = ("clean", "phone", "phone_noisy")
# Transcribe language configurations to compare.
LANG_CONFIGS = {
    "en-IN": {"LanguageCode": "en-IN"},
    "hi-IN": {"LanguageCode": "hi-IN"},
    "auto": {"IdentifyMultipleLanguages": True, "LanguageOptions": ["en-IN", "hi-IN"]},
}
VOICES = (("Kajal", "neural"), ("Arjun", "generative"))  # bilingual en-IN / hi-IN voices
SR = 16000
NOISE_SNR_DB = 10.0


FILLER = {
    "help", "please", "kijiye", "need", "since", "about", "hour", "ago", "minutes", "father",
    "papa", "mere", "meri", "mera", "unka", "unki", "unke", "they", "their", "with", "that",
    "this", "have", "been", "from", "after", "into", "some", "just", "very", "rahi", "raha",
    "rahe", "gaya", "gayi", "hain", "hai", "says", "feels", "cannot", "properly", "side",
}


def symptom_words(text: str) -> list[str]:
    """Content words (>= 4 letters, not filler) whose recall measures how much of the red-flag
    description survives transcription; language-neutral, so EN and Hinglish are comparable."""
    words = re.findall(r"[a-z]+", text.lower())
    return sorted({w for w in words if len(w) >= 4 and w not in FILLER})


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


# ------------------------------------------------------------------ build
def _field(text: str, label: str) -> str | None:
    m = re.search(label + r" is ([^\n.]+)", text)
    return m.group(1).strip() if m else None


def spoken_dob(iso_or_text: str) -> str:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", iso_or_text)
    if not m:
        return iso_or_text
    y, mo, d = m.groups()
    return f"{int(d)} {MONTHS[int(mo) - 1]} {y}"


def digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def build(limit: int | None = None) -> list[dict]:
    from domains.hospital_in.taskgen import REDFLAGS

    tasks = json.loads((ROOT / "domains/hospital_in/data/tasks_generated.json").read_text(encoding="utf-8"))
    items: list[dict] = []
    seen: set[str] = set()
    for t in tasks:
        if not t["id"].endswith("-en"):
            continue
        known = t["user_scenario"]["instructions"]["known_info"] or ""
        name = _field(known, "Your name")
        uhid = _field(known, "Your UHID")
        dob = _field(known, "Your date of birth")
        phone = _field(known, r"Your (?:registered )?mobile number")
        if not (name and uhid and dob) or uhid in seen:
            continue
        seen.add(uhid)
        n = len(items) // 2
        ph = digits(phone)[-10:] if phone else None
        en = f"Hello, my name is {name}. My UHID is {uhid}, and my date of birth is {dob}."
        hi = f"Haan ji, mera naam {name} hai. Mera UHID {uhid} hai, aur date of birth {dob} hai."
        if ph:
            sp = f"{ph[:5]} {ph[5:]}"
            en += f" My mobile number is {sp}."
            hi += f" Mobile number {sp} hai."
        ent = {"name": name, "uhid": digits(uhid), "dob": dob, "phone": ph}
        for lang, text in (("en", en), ("hinglish", hi)):
            items.append({"id": f"id-{n:03d}-{lang}", "kind": "identity", "lang": lang,
                          "text": text, "entities": ent})
    rel = {"en": "my father", "hi": "Papa"}
    pron = {"pron": "he", "Pron": "He", "pos": "his", "rel": "my father", "rel_hi": "Papa"}
    for rf in REDFLAGS:
        en = "I need help. " + rf.flag.format(**pron).capitalize() + "."
        hi = "Please help kijiye. " + rf.hi.format(**pron).capitalize() + "."
        for v in range(len(VOICES)):  # every red flag in both voices: more samples per cell
            for lang, text in (("en", en), ("hinglish", hi)):
                items.append({"id": f"rf-{rf.key}-v{v}-{lang}", "kind": "redflag", "lang": lang,
                              "text": text, "voice_idx": v,
                              "entities": {"redflag": rf.key, "keywords": symptom_words(text)}})
    del rel
    if limit:
        ids = [i for i in items if i["kind"] == "identity"][: 2 * limit]
        items = ids + [i for i in items if i["kind"] == "redflag"]
    return items


# ------------------------------------------------------------------ audio helpers
def write_wav(path: Path, pcm: bytes, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    return x, sr


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """FFT-domain resampling (band-limited); fine for offline speech."""
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x)
    keep = n_out // 2 + 1
    Y = np.zeros(keep, dtype=complex)
    m = min(keep, len(X))
    Y[:m] = X[:m]
    return np.fft.irfft(Y, n_out) * (n_out / len(x))


def bandpass(x: np.ndarray, sr: int, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if n > 1 else 1.0
    return np.fft.irfft(X / np.sqrt(f), n)


def phone_channel(x16: np.ndarray, noisy: bool, seed: int, snr_db: float = NOISE_SNR_DB) -> np.ndarray:
    """16 kHz speech -> 8 kHz telephone-band mu-law speech (decoded back to PCM16)."""
    x = resample(x16, SR, 8000)
    if noisy:
        rng = np.random.default_rng(seed)
        nz = pink_noise(len(x), rng)
        ps = np.mean(x**2) + 1e-9
        pn = np.mean(nz**2) + 1e-9
        x = x + nz * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    x = bandpass(x, 8000) * 0.9
    pcm = np.clip(x, -32768, 32767).astype(np.int16).tobytes()
    return np.frombuffer(audioop.ulaw2lin(audioop.lin2ulaw(pcm, 2), 2), dtype=np.int16).astype(np.float64)


# ------------------------------------------------------------------ AWS steps
def session():
    import boto3
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return boto3.Session(region_name=REGION)


def synth(items: list[dict]) -> None:
    polly = session().client("polly")
    chars = 0
    for k, it in enumerate(items):
        out = OUT / "audio" / "clean" / f"{it['id']}.wav"
        if out.exists():
            continue
        voice, engine = VOICES[it.get("voice_idx", k % len(VOICES))]
        lang = "hi-IN" if it["lang"] == "hinglish" else "en-IN"
        r = polly.synthesize_speech(Text=it["text"], OutputFormat="pcm", SampleRate=str(SR),
                                    VoiceId=voice, Engine=engine, LanguageCode=lang)
        write_wav(out, r["AudioStream"].read(), SR)
        it["voice"] = f"{voice}/{engine}/{lang}"
        chars += len(it["text"])
    print(f"synthesised; {chars} characters sent to Polly")


def channel(items: list[dict]) -> None:
    for k, it in enumerate(items):
        x, sr = read_wav(OUT / "audio" / "clean" / f"{it['id']}.wav")
        for cond, noisy in (("phone", False), ("phone_noisy", True)):
            y = phone_channel(x, noisy, seed=k)
            write_wav(OUT / "audio" / cond / f"{it['id']}.wav",
                      np.clip(y, -32768, 32767).astype(np.int16).tobytes(), 8000)
    print("phone channel applied")


def bucket_name(sess) -> str:
    acct = sess.client("sts").get_caller_identity()["Account"]
    return f"carecall-bench-hearing-{acct}"


def ensure_bucket(sess) -> str:
    s3 = sess.client("s3")
    b = bucket_name(sess)
    try:
        s3.head_bucket(Bucket=b)
    except Exception:  # noqa: BLE001
        s3.create_bucket(Bucket=b)
        s3.put_bucket_tagging(Bucket=b, Tagging={"TagSet": TAG})
        s3.put_public_access_block(Bucket=b, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    return b


def transcribe(items: list[dict]) -> None:
    sess = session()
    s3, tr = sess.client("s3"), sess.client("transcribe")
    b = ensure_bucket(sess)
    jobs = []
    for cond in CONDITIONS:
        for it in items:
            key = f"audio/{cond}/{it['id']}.wav"
            s3.upload_file(str(OUT / key), b, key)
            for cfg_name, cfg in LANG_CONFIGS.items():
                out = OUT / "transcripts" / cond / cfg_name / f"{it['id']}.json"
                if out.exists():
                    continue
                job = f"ccb-{cond}-{cfg_name}-{it['id']}".replace("_", "-")[:200]
                try:
                    tr.start_transcription_job(TranscriptionJobName=job, Media={"MediaFileUri": f"s3://{b}/{key}"},
                                               MediaFormat="wav", Tags=TAG, **cfg)
                except tr.exceptions.ConflictException:
                    pass  # already started earlier; collect it below
                jobs.append((job, out))
    print(f"{len(jobs)} transcription jobs started/collecting")
    pending = dict(jobs)
    while pending:
        for job, out in list(pending.items()):
            st = tr.get_transcription_job(TranscriptionJobName=job)["TranscriptionJob"]
            s = st["TranscriptionJobStatus"]
            if s == "COMPLETED":
                import urllib.request

                data = json.loads(urllib.request.urlopen(st["Transcript"]["TranscriptFileUri"]).read())
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"text": data["results"]["transcripts"][0]["transcript"],
                                           "language": st.get("LanguageCode")}, ensure_ascii=False),
                               encoding="utf-8")
                pending.pop(job)
            elif s == "FAILED":
                print(f"  failed: {job}: {st.get('FailureReason')}")
                pending.pop(job)
        if pending:
            print(f"  {len(pending)} jobs pending...")
            time.sleep(15)


# ------------------------------------------------------------------ scoring
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
HI_NUM = {"शून्य": "0", "ज़ीरो": "0", "जीरो": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4",
          "पाँच": "5", "पांच": "5", "छह": "6", "छः": "6", "सात": "7", "आठ": "8", "नौ": "9",
          "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
          "six": "6", "seven": "7", "eight": "8", "nine": "9", "double": "", "triple": ""}


def normalise(text: str) -> str:
    t = text.translate(DEV_DIGITS).lower()
    for w, d in HI_NUM.items():
        t = re.sub(rf"(?<![\w]){re.escape(w)}(?![\w])", d, t)
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        t = transliterate(t, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
    except ImportError:
        pass
    return t


def digit_stream(t: str) -> str:
    return re.sub(r"\D", "", t)


JUDGE = "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"
JUDGE_PROMPT = """You check a speech-to-text transcript of a caller to a hospital front desk.
The transcript may be in Latin or Devanagari script, or a mix; judge meaning, not spelling or
script (e.g. "फेस" = "face", "राघवेंद्र" = "Raghavendra"). Treat transliteration variants of a
name as correct; a different name is wrong.

Reference (what the caller said): {ref}
Transcript: {hyp}

{ask}
Reply with only a JSON object, no prose."""
ASK_ID = (
    'Answer {"name": true/false, "dob": true/false}: is the caller\'s full name recoverable from '
    "the transcript, and is the full date of birth (day, month, year) recoverable?"
)
ASK_RF = (
    'Answer {"recognisable": true/false, "coverage": 0.0-1.0}: would a front-desk agent reading '
    "only this transcript recognise the same medical emergency (same symptoms), and what fraction "
    "of the reference's symptom details survive?"
)


def judge_transcript(item: dict, hyp: str) -> dict | None:
    """LLM check of names / dates / symptoms that survive transcription, script-independent."""
    import litellm

    ask = ASK_ID if item["kind"] == "identity" else ASK_RF
    msg = JUDGE_PROMPT.format(ref=item["text"], hyp=hyp or "(empty)", ask=ask)
    for _ in range(2):
        try:
            r = litellm.completion(model=JUDGE, messages=[{"role": "user", "content": msg}],
                                   temperature=0, max_tokens=200)
        except Exception as e:  # noqa: BLE001
            print(f"  judge failed: {str(e)[:100]}")
            continue
        txt = r.choices[0].message.content or ""
        s, e_ = txt.find("{"), txt.rfind("}")
        try:
            return json.loads(txt[s : e_ + 1])
        except json.JSONDecodeError:
            continue
    return None


def score() -> dict:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import litellm

    litellm.suppress_debug_info = True
    items = json.loads((OUT / "items.json").read_text(encoding="utf-8"))
    res: dict = {}
    for cond in CONDITIONS:
        for cfg in LANG_CONFIGS:
            for lang in ("en", "hinglish"):
                cells: dict[str, list] = {"uhid": [], "phone": [], "name": [], "dob": [],
                                          "redflag_recognisable": [], "redflag_coverage": []}
                for it in items:
                    if it["lang"] != lang:
                        continue
                    p = OUT / "transcripts" / cond / cfg / f"{it['id']}.json"
                    if not p.exists():
                        continue
                    rec = json.loads(p.read_text(encoding="utf-8"))
                    if "judge" not in rec:  # cached per transcript
                        rec["judge"] = judge_transcript(it, rec["text"])
                        p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                    j = rec["judge"] or {}
                    ds = digit_stream(normalise(rec["text"]))
                    e = it["entities"]
                    if it["kind"] == "identity":
                        cells["uhid"].append(e["uhid"] in ds)
                        if e.get("phone"):
                            cells["phone"].append(e["phone"] in ds)
                        if "name" in j:
                            cells["name"].append(bool(j["name"]))
                        if "dob" in j:
                            cells["dob"].append(bool(j["dob"]))
                    else:
                        if "recognisable" in j:
                            cells["redflag_recognisable"].append(bool(j["recognisable"]))
                        if "coverage" in j:
                            cells["redflag_coverage"].append(float(j["coverage"]))
                res[f"{cond}|{cfg}|{lang}"] = {
                    k: (round(sum(v) / len(v), 3), len(v)) for k, v in cells.items() if v}
    (OUT / "score.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for k, v in res.items():
        print(k, v)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["build", "synth", "channel", "transcribe", "score", "all"])
    ap.add_argument("--limit", type=int, help="identity items per language (default: all)")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    items_path = OUT / "items.json"
    if args.step in ("build", "all"):
        items = build(args.limit)
        items_path.write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{len(items)} items ({sum(i['kind'] == 'identity' for i in items)} identity, "
              f"{sum(i['kind'] == 'redflag' for i in items)} red-flag)")
        if args.step == "build":
            return 0
    items = json.loads(items_path.read_text(encoding="utf-8"))
    if args.step in ("synth", "all"):
        synth(items)
        items_path.write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.step in ("channel", "all"):
        channel(items)
    if args.step in ("transcribe", "all"):
        transcribe(items)
    if args.step in ("score", "all"):
        score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
