#!/usr/bin/env python3
"""pipeline.py [steps] --target T.wav --reference R.wav --work DIR [--key ELEVENLABS_KEY] [--names Richard,Liron]

Whole-track bleed-artifact removal with a human review tier.  Steps (each cached in --work):
  analyse   islands of TARGET + level/duration/ref-activity/waveform-ncc/own_dist   -> rows.json, segs/
  scribe    ElevenLabs Scribe on REFERENCE (once) and on every candidate island     -> ref_scribe.json, segs_json/
  score     evidence points -> p(artifact) -> tier remove / keep / review            -> scored.json, overlaps.csv
  render    mute the remove tier (10 ms fades)                                        -> <target>_debleed.wav
  review    clips + static HTML page for the review tier                              -> review/index.html
  learn     --decisions FILE: distil the reviewer's calls into rules, re-tier the undecided  -> learned.json
  apply     --decisions FILE (+ learned.json if present)                              -> <target>_final.wav
"""
import argparse, csv, difflib, glob, hashlib, html, json, math, os, re, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debleed import decode, envelope, analyse, classify, render, mmss

BC = {"yeah", "yes", "yep", "yup", "right", "mm", "mhm", "mmhmm", "uhhuh", "okay", "ok", "sure", "exactly", "no", "hmm", "hm",
      "true", "wow", "oh", "um", "uh", "mhmm", "huh", "totally", "absolutely", "cool", "nice", "i", "see", "interesting", "yea"}
norm = lambda s: re.sub(r"[^a-z']", "", s.lower())
J = lambda p: json.load(open(p, encoding="utf-8"))
def dump(o, p): json.dump(o, open(p, "w", encoding="utf-8"), ensure_ascii=False)
def ff(*args): subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)

def scribe(key, path):
    import requests
    r = requests.post("https://api.elevenlabs.io/v1/speech-to-text", headers={"xi-api-key": key},
                      data={"model_id": "scribe_v1", "timestamps_granularity": "word", "diarize": "false"},
                      files={"file": open(path, "rb")}, timeout=600)
    r.raise_for_status(); return r.json()

def spectral(T, rows, sr=16000):
    """Per-island timbre: bleed arrives via speakers -> room -> mic and loses the low-frequency
    voicing a close mic gives the speaker's own voice, so it sounds whispery. centroid (Hz),
    tilt = 2-4 kHz vs 200-1000 Hz energy (dB), voicing = max normalised autocorrelation over
    pitch lags 80-400 Hz. Medians over the frames within 20 dB of the island's peak."""
    n, h = 512, 160; win = np.hanning(n); f = np.fft.rfftfreq(n, 1 / sr)
    b1, b2 = (f >= 2000) & (f < 4000), (f >= 200) & (f < 1000)
    for r in rows:
        x = T[int(r["t0"] * sr):int(r["t1"] * sr)]
        if len(x) < n: x = np.pad(x, (0, n - len(x)))
        fr = np.lib.stride_tricks.sliding_window_view(x, n)[::h] * win
        e = (fr ** 2).mean(1); fr = fr[e > e.max() * 0.01]
        S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        ac = np.fft.irfft(S, axis=1)[:, :n]; ac = ac / (ac[:, :1] + 1e-12)
        r["centroid"] = round(float(np.median((S * f).sum(1) / (S.sum(1) + 1e-12))))
        r["tilt"] = round(float(np.median(10 * np.log10((S[:, b1].sum(1) + 1e-12) / (S[:, b2].sum(1) + 1e-12)))), 1)
        r["voicing"] = round(float(np.median(ac[:, int(sr / 400):int(sr / 80)].max(1))), 2)
    wh = [(r["t0"], r["t1"]) for r in rows if r["centroid"] >= 1500 or r["tilt"] >= -5]
    for r in rows:   # whispery islands within 1.5 s of this one (context: a blip inside a bleed stretch)
        r["nbr_whisper"] = sum(1 for t0, t1 in wh if t0 != r["t0"] and t0 < r["t1"] + 1.5 and t1 > r["t0"] - 1.5)

def step_analyse(a):
    T, R = decode(a.target), decode(a.reference)
    tdb, rdb = envelope(T), envelope(R); n = min(len(tdb), len(rdb)); tdb, rdb = tdb[:n], rdb[:n]
    s90, rows = analyse(T, R, tdb, rdb); rows = classify(rows, s90); spectral(T, rows)
    own = [(r["t0"], r["t1"]) for r in rows if r["peak"] > -16 and r["dur"] >= 0.8]
    os.makedirs(a.work + "/segs", exist_ok=True); cand = []
    for r in rows:
        r["own_dist"] = round(min([min(abs(r["t0"] - t1), abs(t0 - r["t1"])) for t0, t1 in own if t0 != r["t0"]] or [999]), 2)
        if r["ract"] >= 0.2 and r["peak"] > -32 and r["dur"] >= 0.15:
            r["id"] = "seg%04d" % len(cand); cand.append(r["id"])
            ff("-ss", "%.3f" % max(0, r["t0"] - 0.15), "-t", "%.3f" % (r["dur"] + 0.3), "-i", a.target, "-ac", "1", "-ar", "16000", "%s/segs/%s.wav" % (a.work, r["id"]))
    dump(dict(s90=s90, rows=rows, cand=cand), a.work + "/rows.json")
    print("analyse: %d islands, %d rule-removed, %d candidates for STT, speech p90 %.1f dBFS" % (len(rows), sum(1 for r in rows if r["why"]), len(cand), s90))

def step_scribe(a):
    key = a.key or os.environ.get("ELEVENLABS_API_KEY") or sys.exit("need --key or ELEVENLABS_API_KEY")
    ref = a.work + "/ref_scribe.json"
    if not os.path.exists(ref):
        mp3 = a.work + "/ref16k.mp3"; ff("-i", a.reference, "-ac", "1", "-ar", "16000", "-b:a", "48k", mp3)
        dump(scribe(key, mp3), ref); print("scribe: reference transcribed")
    os.makedirs(a.work + "/segs_json", exist_ok=True)
    for sid in J(a.work + "/rows.json")["cand"]:
        out = "%s/segs_json/%s.json" % (a.work, sid)
        if not os.path.exists(out): dump(scribe(key, "%s/segs/%s.wav" % (a.work, sid)), out)
    print("scribe: %d candidate segments transcribed" % len(glob.glob(a.work + "/segs_json/*.json")))

def step_score(a):
    d = J(a.work + "/rows.json"); rows, s90 = d["rows"], d["s90"]
    L = [w for w in J(a.work + "/ref_scribe.json")["words"] if w["type"] == "word"]
    seg = {os.path.basename(f)[:-5]: [norm(w["text"]) for w in J(f).get("words", []) if w["type"] == "word" and norm(w["text"])]
           for f in glob.glob(a.work + "/segs_json/*.json")}
    lwords = lambda t0, t1: [norm(w["text"]) for w in L if t0 - 0.75 <= w["start"] <= t1 - 0.05]
    def match(rw, lw):
        return sum(1 for w in rw if w in lw or any(difflib.SequenceMatcher(None, w, x).ratio() >= 0.8 for x in lw)) / len(rw) if rw else 0.0
    for r in rows:
        if r["ract"] < 0.2:   # reference not active: not scored, but an island the hard rules flagged must not fall through the cracks
            if r["why"]: r.update(rel=round(r["peak"] - s90, 1), words="", ref_words="", lex=0.0, score=None, p=0.9 if r["peak"] <= -30 else 0.5, evidence="hard rule: %s; reference quiet so not scored" % r["why"])
            r["tier"] = "n/a" if not r["why"] else "remove" if r["peak"] <= -30 else "review"; continue
        rel = r["peak"] - s90; rw = seg.get(r.get("id", ""), None); lw = lwords(r["t0"], r["t1"]); m = match(rw, lw) if rw else 0.0
        ev = []; s = 0
        if r["ncc"] >= 0.15: s += 3; ev.append("waveform=%s ncc %.2f" % (a.ref_name, r["ncc"]))
        elif r["ncc"] >= 0.10: s += 1.5; ev.append("ncc %.2f" % r["ncc"])
        if rw is None: s += 2; ev.append("too quiet to transcribe")
        elif not rw and rel <= -6: s += 2; ev.append("no words")
        elif not rw: ev.append("loud but no words (laugh / non-verbal?)")
        elif m >= 0.6: s += 3; ev.append("echoes %s: %s" % (a.ref_name, " ".join(rw)))
        elif m >= 0.3: s += 1; ev.append("partly echoes %s (%d%%)" % (a.ref_name, m * 100))
        if rel <= -18: s += 3; ev.append("%.0f dB under speech" % rel)
        elif rel <= -10: s += 2; ev.append("%.0f dB under speech" % rel)
        elif rel <= -6: s += 1; ev.append("%.0f dB under speech" % rel)
        if r["dur"] <= 0.4: s += 1
        if r["own_dist"] > 1.5: s += 1; ev.append("far from own speech")
        whisper = r["centroid"] >= 1500 or r["tilt"] >= -5; voiced = r["centroid"] < 700 and r["tilt"] < -15
        real = bool(rw) and len(rw) >= 3 and rel > -6 and m < 0.3
        if whisper: s += 3; ev.append("whispery (centroid %d Hz, tilt %+.0f dB)" % (r["centroid"], r["tilt"]))
        elif voiced: s -= 1.5; ev.append("voiced / close-mic")
        if not whisper and not rw and rel > -6 and r["dur"] < 0.35: s += 3; ev.append("voiced blip, no words")
        if voiced and not rw and r["dur"] >= 1.0: s += 2; ev.append("long voiced non-verbal (%.1fs), no words" % r["dur"])
        if r["nbr_whisper"] >= 2 and not real: s += 2; ev.append("inside a bleed stretch (%d whispery islands within 1.5 s)" % r["nbr_whisper"])
        if rw and m < 0.3:
            if all(w in BC for w in rw):
                if rel <= -6: s += 1.5; ev.append("quiet backchannel under %s: %s" % (a.ref_name, " ".join(rw)))
                else: s -= 3; ev.append("backchannel at speech level: %s" % " ".join(rw))
            elif real: s -= 4; ev.append("real speech: %s" % " ".join(rw))
            elif rel > -6 and not whisper: s -= 3; ev.append("words at speech level: %s" % " ".join(rw))
            elif rel > -3 and whisper: s -= 2.5; ev.append("loud words despite whispery timbre: %s" % " ".join(rw))
            elif rel <= -10 and len(rw) <= 2: s += 1; ev.append("quiet mutter under %s: %s" % (a.ref_name, " ".join(rw)))
            else: s -= 1; ev.append("words: %s" % " ".join(rw))
        if r["own_dist"] <= 0.3: s -= 2; ev.append("adjacent to own speech")
        if r["why"] == "sustained low level under reference": s += 3; ev.append("gate stuck open %.0fs" % r["dur"])
        if r["peak"] <= -40: s = max(s, 6); ev.append("inaudible")
        p = 1 / (1 + math.exp(-(s - 2.5)))
        r.update(rel=round(rel, 1), words=" ".join(rw) if rw else "", ref_words=" ".join(lw), lex=round(m, 2), score=s, p=round(p, 2), evidence="; ".join(ev))
        r["tier"] = "remove" if p >= a.remove_p else "keep" if p <= a.keep_p else "review"
    dump(dict(s90=s90, rows=rows), a.work + "/scored.json")
    with open(a.work + "/overlaps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["start", "end", "dur_s", "peak_dbfs", "vs_speech_db", "centroid_hz", "tilt_db", "ncc", "target_words", "ref_words_before", "lexical_match", "p_artifact", "tier", "evidence"])
        for r in rows:
            if r["tier"] != "n/a": w.writerow([mmss(r["t0"]), mmss(r["t1"]), "%.2f" % r["dur"], "%.1f" % r["peak"], r["rel"], r["centroid"], r["tilt"], "%.2f" % r["ncc"], r["words"], r["ref_words"], r["lex"], r["p"], r["tier"], r["evidence"]])
    from collections import Counter
    C = Counter(r["tier"] for r in rows); tot = lambda t: sum(r["dur"] for r in rows if r["tier"] == t)
    print("score: remove %d (%.0fs)  keep %d (%.0fs)  review %d (%.0fs)" % (C["remove"], tot("remove"), C["keep"], tot("keep"), C["review"], tot("review")))

def sr_of(p): return int(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate", "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip())

def step_render(a, decisions=None):
    rows = J(a.work + "/scored.json")["rows"]
    decisions = decisions or {}   # an explicit reviewer decision overrides the tier, whatever tier it is
    cuts = [(r["t0"], r["t1"]) for r in rows if decisions.get(mmss(r["t0"]), "remove" if r["tier"] == "remove" else "keep") == "remove"]
    out = a.out or os.path.splitext(a.target)[0] + ("_final.wav" if decisions else "_debleed.wav")
    render(a.target, out, cuts, sr_of(a.target))
    print("render: %d cuts, %.1f s muted -> %s" % (len(cuts), sum(b - x for x, b in cuts), out))

def step_review(a):
    rows = J(a.work + "/scored.json")["rows"]; done = read_decisions(a.decisions) if a.decisions else {}
    rv = [r for r in rows if r["tier"] == "review" and mmss(r["t0"]) not in done]   # --decisions: skip rows already called
    d = a.work + "/review"; os.makedirs(d + "/clips", exist_ok=True); cards = []
    for i, r in enumerate(rv):
        rid = "rv%02d" % i; t0 = max(0, r["t0"] - 1); dur = r["dur"] + 2
        ff("-ss", "%.3f" % t0, "-t", "%.3f" % dur, "-i", a.target, "-ac", "1", "-b:a", "96k", "%s/clips/%s_target.mp3" % (d, rid))
        cards.append('<tr id="{rid}"><td class="t">{t0}–{t1}<br><span class="m">{dur:.2f} s · {peak:.0f} dBFS ({rel:+.0f} dB)</span></td>'
                     '<td><audio controls preload="none" src="clips/{rid}_target.mp3"></audio><div class="w">heard: {rw}</div></td>'
                     '<td><div class="w">{lw}</div></td>'
                     '<td class="ev">p={p:.2f}<br>{ev}</td><td class="dec"><label><input type="radio" name="{rid}" value="remove"> remove</label>'
                     '<label><input type="radio" name="{rid}" value="keep"> keep</label></td></tr>'.format(
                         rid=rid, t0=mmss(r["t0"]), t1=mmss(r["t1"]), dur=r["dur"], peak=r["peak"], rel=r["rel"], rw=html.escape(r["words"] or "—"),
                         lw=html.escape(r["ref_words"] or "—"), p=r["p"], ev=html.escape(r["evidence"])))
    from collections import Counter
    C = Counter(r["tier"] for r in rows); tot = lambda t: "%.0f" % sum(r["dur"] for r in rows if r["tier"] == t); N = sum(C[t] for t in ("remove", "keep", "review"))
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_template.html"), encoding="utf-8").read()
    for k, v in {"@N_REVIEW@": len(rv), "@N@": N, "@N_REMOVE@": C["remove"], "@S_REMOVE@": tot("remove"), "@N_KEEP@": C["keep"], "@S_KEEP@": tot("keep"),
                 "@S_REVIEW@": tot("review"), "@ROWS@": "\n".join(cards), "@TARGET@": a.target_name, "@REF@": a.ref_name}.items(): page = page.replace(k, str(v))
    page = page.replace('K="overlap-review-v1"', 'K="overlap-review-%s"' % hashlib.md5("".join(mmss(r["t0"]) for r in rv).encode()).hexdigest()[:8])   # storage key per row set: a rebuilt page never inherits stale radio state
    open(d + "/index.html", "w", encoding="utf-8", newline="\n").write(page); print("review: %d segments -> %s/index.html" % (len(rv), d))

def read_decisions(path):
    dec = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*(\d+:\d\d\.\d\d)\S*\s+(remove|keep)", line)
        if m: dec[m.group(1)] = m.group(2)
    return dec

# features the reviewer's decisions are distilled over: (name, extractor, unit for the printed rule)
FEATS = [("vs_speech_db", lambda r: r["rel"], "dB"), ("centroid_hz", lambda r: r["centroid"], "Hz"), ("tilt_db", lambda r: r["tilt"], "dB"), ("voicing", lambda r: r["voicing"], ""), ("nbr_whisper", lambda r: r["nbr_whisper"], ""), ("ncc", lambda r: r["ncc"], ""), ("dur_s", lambda r: r["dur"], "s"),
         ("lexical_match", lambda r: r["lex"], ""), ("n_words", lambda r: len(r["words"].split()), ""),
         ("own_dist_s", lambda r: min(r["own_dist"], 10), "s"), ("peak_dbfs", lambda r: r["peak"], "dBFS"), ("p_artifact", lambda r: r["p"], "")]

def step_learn(a):
    """Distil the reviewer's decisions into (1) the single-feature threshold rules that best separate remove from keep,
    (2) a small ridge-logistic model over all features, used to re-tier the undecided review rows.  -> learned.json"""
    import numpy as np
    rows = J(a.work + "/scored.json")["rows"]; dec = read_decisions(a.decisions)
    rv = [r for r in rows if r["tier"] == "review"]; lab = [(r, dec.get(mmss(r["t0"]))) for r in rv]
    done = [(r, y) for r, y in lab if y]; todo = [r for r, y in lab if not y]
    if not done: sys.exit("learn: no decisions match review rows")
    X = np.array([[f(r) for _, f, _ in FEATS] for r, _ in done]); y = np.array([1.0 if v == "remove" else 0.0 for _, v in done])
    print("learn: %d decided (%d remove, %d keep), %d undecided" % (len(done), int(y.sum()), int(len(y) - y.sum()), len(todo)))
    rules = []
    for j, (name, _, unit) in enumerate(FEATS):                       # exhaustive single-threshold search
        for t in sorted(set(X[:, j])):
            for op in ("<=", ">"):
                m = X[:, j] <= t if op == "<=" else X[:, j] > t
                if m.sum() < 2: continue
                pur = float(y[m].mean()); side = "remove" if pur >= 0.5 else "keep"; acc = max(pur, 1 - pur)
                rules.append(dict(rule="%s %s %.2f%s" % (name, op, t, unit), verdict=side, n=int(m.sum()), purity=round(acc, 2), covered=int(m.sum() * acc)))
    rules.sort(key=lambda d: (0 if d["purity"] >= 0.9 else 1, -d["covered"], -d["purity"])); top = rules[:8]
    print("learn: cleanest single-feature rules from your decisions:")
    for d in top: print("   %-32s -> %-6s  %d/%d agree" % (d["rule"], d["verdict"], d["covered"], d["n"]))
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; w = np.zeros(Z.shape[1]); b = 0.0    # ridge logistic, GD
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(Z @ w + b))); g = p - y
        w -= 0.1 * (Z.T @ g / len(y) + 0.05 * w); b -= 0.1 * g.mean()
    fit = ((1 / (1 + np.exp(-(Z @ w + b))) >= 0.5) == (y == 1)).mean()
    print("learn: model fits %.0f%% of your decisions; feature weights (+ = artifact):" % (fit * 100))
    for (name, _, _), wi in sorted(zip(FEATS, w), key=lambda t: -abs(t[1])): print("   %+.2f  %s" % (wi, name))
    out = []
    for r in todo:
        z = (np.array([f(r) for _, f, _ in FEATS]) - mu) / sd; p = float(1 / (1 + np.exp(-(z @ w + b))))
        r["p_learned"] = round(p, 2); r["tier_learned"] = "remove" if p >= 0.5 else "keep"; out.append(r)
    from collections import Counter
    print("learn: undecided re-tiered -> %s" % dict(Counter(r["tier_learned"] for r in out)))
    for r in sorted(out, key=lambda r: r["p_learned"]): print("   %s  p_learned %.2f -> %-6s  (was p %.2f)  %s" % (mmss(r["t0"]), r["p_learned"], r["tier_learned"], r["p"], r["evidence"][:70]))
    dump(dict(decisions=dec, rules=top, weights=dict(zip([n for n, _, _ in FEATS], map(float, w))), bias=float(b), mu=mu.tolist(), sd=sd.tolist(), fit=float(fit),
              undecided=[dict(start=mmss(r["t0"]), p_learned=r["p_learned"], tier=r["tier_learned"]) for r in out]), a.work + "/learned.json")

def step_apply(a):
    dec = read_decisions(a.decisions)
    if os.path.exists(a.work + "/learned.json") and not a.no_learned:      # reviewer's explicit calls + learned calls for the rest
        for u in J(a.work + "/learned.json")["undecided"]: dec.setdefault(u["start"], u["tier"])
    print("apply: %d decisions (%d remove)" % (len(dec), sum(v == "remove" for v in dec.values()))); step_render(a, dec)

def step_blind(a):
    """Blind test of the scorer. Without --decisions: sample --n islands the reviewer has never seen (--exclude prior decisions),
    build a page that shows ONLY the audio and the words (no p, no tier, no evidence) and seal the scorer's calls in work/blind_key.json.
    With --decisions: grade the reviewer's calls against the sealed key."""
    import random
    key_path = a.work + "/blind_key.json"
    if a.decisions:
        key = J(key_path); dec = read_decisions(a.decisions); agree = conf = confagree = 0
        print("blind test: %d sealed calls, %d reviewer decisions" % (len(key), len(dec)))
        print("   %-9s %-6s %-6s %-5s %s" % ("where", "you", "algo", "p", "evidence"))
        for k in sorted(key, key=lambda k: key[k]["t0"]):
            r = key[k]; you = dec.get(r["start"]); algo = r["tier"]
            if you is None: continue
            ok = you == algo; agree += ok
            if algo != "review": conf += 1; confagree += ok
            print("   %-9s %-6s %-6s %.2f  %s  %s" % (r["start"], you, algo, r["p"], "ok " if ok else "XX " if algo != "review" else "-- ", r["evidence"][:90]))
        n = sum(1 for k in key if key[k]["start"] in dec)
        print("blind test: %d/%d agree overall; on the %d confident calls (not review tier) %d/%d agree" % (agree, n, conf, confagree, conf))
        return
    rows = [r for r in J(a.work + "/scored.json")["rows"] if "p" in r]   # overlap islands only (the rest were never scored)
    seen = set(read_decisions(a.exclude)) if a.exclude else set()
    pool = [r for r in rows if mmss(r["t0"]) not in seen]
    strata = [("remove", lambda r: r["p"] >= 0.95, 0.4), ("remove", lambda r: a.remove_p <= r["p"] < 0.95, 0.2), ("review", lambda r: r["tier"] == "review", 0.2), ("keep", lambda r: r["tier"] == "keep", 0.2)]
    rng = random.Random(a.seed); pick = []
    for _, f, frac in strata:
        c = [r for r in pool if f(r) and r not in pick]; rng.shuffle(c); pick += c[:max(1, round(a.n * frac))]
    pick = sorted(pick[:a.n], key=lambda r: r["t0"])
    d = a.work + "/blind"; os.makedirs(d + "/clips", exist_ok=True); cards = []; key = {}
    for i, r in enumerate(pick):
        rid = "bt%02d" % i; t0 = max(0, r["t0"] - 1); dur = r["dur"] + 2
        ff("-ss", "%.3f" % t0, "-t", "%.3f" % dur, "-i", a.target, "-ac", "1", "-b:a", "96k", "%s/clips/%s_target.mp3" % (d, rid))
        key[rid] = dict(start=mmss(r["t0"]), t0=r["t0"], p=r["p"], tier=r["tier"], evidence=r["evidence"], centroid=r["centroid"], tilt=r["tilt"], rel=r["rel"], words=r["words"])
        cards.append('<tr id="{rid}"><td class="t">{t0}–{t1}<br><span class="m">{dur:.2f} s</span></td>'
                     '<td><audio controls preload="none" src="clips/{rid}_target.mp3"></audio><div class="w">heard: {rw}</div></td>'
                     '<td><div class="w">{lw}</div></td><td class="dec"><label><input type="radio" name="{rid}" value="remove"> remove</label>'
                     '<label><input type="radio" name="{rid}" value="keep"> keep</label></td></tr>'.format(
                         rid=rid, t0=mmss(r["t0"]), t1=mmss(r["t1"]), dur=r["dur"], rw=html.escape(r["words"] or "—"), lw=html.escape(r["ref_words"] or "—")))
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_template.html"), encoding="utf-8").read()
    page = re.sub(r"<h1>.*?</h1>\s*<p class=\"sub\">.*?</p>\s*<div class=\"stats\">.*?</div>\s*", (
        "<h1>@TARGET@ blind test — %d segments</h1><p class=\"sub\">Same format as the review page, but the scorer's opinion is hidden: each row is "
        "@TARGET@'s track alone (1 s of context each side) plus what ElevenLabs heard and what @REF@ said just before. Mark every row, Copy decisions, paste back — "
        "the sealed predictions are then graded against yours.</p>" % len(pick)), page, count=1, flags=re.S)
    page = page.replace("<th>Evidence</th>", "").replace("overlap-review-v1", "overlap-blind-v1")
    page = re.sub(r"<p class=\"m\">Full scored table.*?</p>", "", page, flags=re.S)
    for k, v in {"@ROWS@": "\n".join(cards), "@TARGET@": a.target_name, "@REF@": a.ref_name}.items(): page = page.replace(k, str(v))
    open(d + "/index.html", "w", encoding="utf-8", newline="\n").write(page); dump(key, key_path)
    from collections import Counter
    print("blind: %d segments -> %s/index.html ; sealed key (tiers %s) -> %s" % (len(pick), d, dict(Counter(k["tier"] for k in key.values())), key_path))

STEPS = {"analyse": step_analyse, "scribe": step_scribe, "score": step_score, "render": step_render, "review": step_review, "learn": step_learn, "apply": step_apply, "blind": step_blind}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", default=["analyse", "scribe", "score", "render", "review"], choices=list(STEPS))
    ap.add_argument("--target", required=True); ap.add_argument("--reference", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--key"); ap.add_argument("--names", default="Target,Reference"); ap.add_argument("--out"); ap.add_argument("--decisions")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--exclude")
    ap.add_argument("--remove-p", type=float, default=0.70); ap.add_argument("--keep-p", type=float, default=0.20); ap.add_argument("--no-learned", action="store_true")
    a = ap.parse_args(); a.target_name, a.ref_name = (a.names.split(",") + ["Reference"])[:2]; os.makedirs(a.work, exist_ok=True)
    for s in a.steps: STEPS[s](a)
