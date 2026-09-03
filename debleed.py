#!/usr/bin/env python3
"""debleed.py  TARGET.wav  REFERENCE.wav  OUT.wav  [--csv report.csv]

Remove the other speaker's bleed/echo from a gated Riverside track.

The problem: TARGET's noise gate opens on the other person's voice coming out of
the speaker's headphones/speakers.  The result is short, speech-shaped blips in
TARGET's pauses that sit exactly where REFERENCE (the other speaker's own track)
is talking, delayed by the network round-trip (150-450 ms here, and it jitters,
so a fixed-delay echo canceller does not work).  Speech isolation keeps them
because they ARE speech.

Method (all automatic, no thresholds tuned by ear):
  1. 10 ms RMS envelopes of both tracks.
  2. Split TARGET into "islands" - the ungated stretches (gaps < 100 ms merged).
  3. For every island, measure: duration, peak level, whether REFERENCE is
     talking in the 100-600 ms before/through it, and the best waveform
     cross-correlation against REFERENCE over that delay range (bleed matches
     the reference waveform; the speaker's own words do not).
  4. Classify.  An island is BLEED if reference is active and any of:
       - it is short (<= 0.6 s) and quiet (peak <= -20 dBFS): gate chatter
       - its waveform correlates with the reference (ncc >= 0.15)
       - it is well under speech level (peak <= speech_p90 - 10 dB) and <= 1.2 s
       - it stays >= 10 dB under speech level (95th pct) for as long as the
         reference talks (ract >= 0.8): a gate stuck open on steady bleed/hum
     An island is KEPT as the speaker's own overlapping speech if it is loud
     (peak > -16 dBFS), sustained (>= 0.8 s) and does NOT match the reference.
  5. Neighbourhood rule: islands (<= 1.5 s) within 1.0 s of a BLEED island, while the
     reference is still talking, that are not clearly own-speech, are removed
     too - a stray word next to a cluster of artifacts is almost always more of
     the same bleed (a gate opening on a louder syllable of the other speaker).
  6. Render: the removed ranges are muted with 10 ms fades.  Nothing else in the
     track is touched; own speech is bit-identical.
"""
import argparse, csv, subprocess, sys
import numpy as np
from scipy.signal import correlate, butter, sosfilt

SR = 16000; HOP = 160; F = HOP / SR          # analysis rate 10 ms

def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768

def envelope(x):
    f = x[:(len(x) // HOP) * HOP].reshape(-1, HOP)
    return 20 * np.log10(np.sqrt((f * f).mean(1)) + 1e-12)

def islands(db, gate=-85.0, merge=10):
    act = db > gate; out = []; i = 0; n = len(db)
    while i < n:
        if act[i]:
            j = i
            while j < n and (act[j] or act[j + 1:j + 1 + merge].any()): j += 1
            out.append((i, j)); i = j
        else: i += 1
    return out

def analyse(T, R, tdb, rdb):
    sos = butter(4, [300, 3400], btype="band", fs=SR, output="sos")
    s90 = float(np.percentile(tdb[tdb > -85], 90))
    rows = []
    for a, b in islands(tdb):
        dur = (b - a) * F
        ref = rdb[max(0, a - 60):max(0, b - 5)]                  # 100-600 ms earlier
        ract = float((ref > -50).mean()) if len(ref) else 0.0
        ncc = 0.0; lag = None
        if ract >= 0.2 and dur >= 0.1:
            x = sosfilt(sos, T[a * HOP:b * HOP])
            y0 = max(0, a * HOP - int(0.7 * SR)); y = sosfilt(sos, R[y0:b * HOP])
            if len(y) > len(x) + 10:
                c = correlate(x, y, mode="valid") / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12)
                k = int(np.argmax(np.abs(c))); ncc = float(abs(c[k])); lag = (a * HOP - (y0 + k)) / SR
        rows.append(dict(a=a, b=b, t0=a * F, t1=b * F, dur=dur, peak=float(tdb[a:b].max()),
                         p95=float(np.percentile(tdb[a:b], 95)),
                         mean=float(tdb[a:b].mean()), ract=ract, ncc=ncc, lag=lag))
    return s90, rows

def classify(rows, s90):
    for r in rows:
        r["why"] = ""
        own = r["peak"] > -16 and r["dur"] >= 0.8 and r["ncc"] < 0.12
        if r["ract"] >= 0.4 and not own:
            if r["dur"] <= 0.6 and r["peak"] <= -20: r["why"] = "gate chatter"
            elif r["ncc"] >= 0.15: r["why"] = "waveform matches reference (ncc %.2f)" % r["ncc"]
            elif r["peak"] <= s90 - 10 and r["dur"] <= 1.2: r["why"] = "quiet, reference talking"
            elif r["p95"] <= s90 - 10 and r["ract"] >= 0.8: r["why"] = "sustained low level under reference"
        r["own"] = own
    # neighbourhood rule
    bleed_t = [(r["t0"], r["t1"]) for r in rows if r["why"]]
    for r in rows:
        if r["why"] or r["own"]: continue
        if r["ract"] < 0.1 and not (r["dur"] <= 0.3 and r["peak"] <= -20): continue
        if r["dur"] > 1.5 or (r["peak"] > -16 and r["dur"] > 1.0): continue
        near = any(abs(r["t0"] - t1) <= 1.0 or abs(t0 - r["t1"]) <= 1.0 for t0, t1 in bleed_t)
        if near: r["why"] = "next to bleed cluster"
    return rows

def render(src, out, cuts, sr_out):
    """Mute cuts (seconds) with 10 ms fades; keep original sample rate/format via ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, dtype="<f4").copy(); fade = int(0.01 * sr_out)
    ramp = np.linspace(1, 0, fade, dtype=np.float32)
    for t0, t1 in cuts:
        a, b = max(0, int(t0 * sr_out) - fade), min(len(x), int(t1 * sr_out) + fade)
        if b - a <= 2 * fade: x[a:b] = 0; continue
        x[a:a + fade] *= ramp; x[a + fade:b - fade] = 0; x[b - fade:b] *= ramp[::-1]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr_out), "-ac", "1", "-i", "-",
                    "-c:a", "pcm_s16le", out], input=x.tobytes(), check=True)

def mmss(t): return "%d:%05.2f" % (t // 60, t % 60)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("reference"); ap.add_argument("out")
    ap.add_argument("--csv"); args = ap.parse_args()
    T, R = decode(args.target), decode(args.reference)
    tdb, rdb = envelope(T), envelope(R); n = min(len(tdb), len(rdb)); tdb, rdb = tdb[:n], rdb[:n]
    s90, rows = analyse(T, R, tdb, rdb); rows = classify(rows, s90)
    cuts = [(r["t0"], r["t1"]) for r in rows if r["why"]]
    sr_out = int(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate", "-of", "csv=p=0",
                                 args.target], capture_output=True, text=True).stdout.strip())
    render(args.target, args.out, cuts, sr_out)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["start", "end", "start_s", "end_s", "dur_s", "peak_dbfs", "ref_active", "ncc", "lag_ms", "reason"])
            for r in rows:
                if r["why"]:
                    w.writerow([mmss(r["t0"]), mmss(r["t1"]), "%.2f" % r["t0"], "%.2f" % r["t1"], "%.2f" % r["dur"],
                                "%.1f" % r["peak"], "%.2f" % r["ract"], "%.3f" % r["ncc"],
                                "" if r["lag"] is None else int(r["lag"] * 1000), r["why"]])
    rem = [r for r in rows if r["why"]]
    from collections import Counter
    print("islands: %d   removed: %d (%.1f s)   speech p90 %.1f dBFS" % (len(rows), len(rem), sum(r["dur"] for r in rem), s90))
    for k, v in Counter(r["why"].split(" (")[0] for r in rem).most_common(): print("  %-32s %d" % (k, v))
    print("wrote", args.out)
