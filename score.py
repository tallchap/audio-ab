"""Score every Richard-over-Liron overlap segment: artifact likelihood from level, duration,
waveform correlation, lexical echo match against Liron's transcript, and proximity to Richard's own speech."""
import json, glob, re, math, difflib, csv
rows=json.load(open("rows.json"))["rows"]; s90=json.load(open("rows.json"))["s90"]
L=[w for w in json.load(open("liron_scribe.json",encoding="utf-8"))["words"] if w["type"]=="word"]
norm=lambda s: re.sub(r"[^a-z']","",s.lower())
BC={"yeah","yes","yep","yup","right","mm","mhm","mmhmm","uhhuh","okay","ok","sure","exactly","no","hmm","hm","true","wow","oh","um","uh","mhmm","huh","totally","absolutely","cool","nice","i","see","interesting","yea"}
seg={}
for f in glob.glob("segs_json/*.json"):
    d=json.load(open(f,encoding="utf-8")); seg[f.split("\\")[-1].split("/")[-1][:-5]]=[norm(w["text"]) for w in d.get("words",[]) if w["type"]=="word" and norm(w["text"])]
def liron_words(t0,t1): return [norm(w["text"]) for w in L if t0-0.75<=w["start"]<=t1-0.05]
def match(rw,lw):
    if not rw: return 0.0
    hit=0
    for w in rw:
        if w in lw or any(difflib.SequenceMatcher(None,w,x).ratio()>=0.8 for x in lw): hit+=1
    return hit/len(rw)
bleed_t=[]
out=[]
for r in rows:
    if r["ract"]<0.2: r["tier"]="n/a"; continue
    rel=r["peak"]-s90; rw=seg.get(r.get("id",""),None); lw=liron_words(r["t0"],r["t1"])
    m=match(rw,lw) if rw else 0.0
    ev=[]; s=0
    if r["ncc"]>=0.15: s+=3; ev.append("waveform=Liron ncc %.2f"%r["ncc"])
    elif r["ncc"]>=0.10: s+=1.5; ev.append("ncc %.2f"%r["ncc"])
    if rw is None: s+=2; ev.append("too quiet to transcribe")
    elif not rw and rel<=-6: s+=2; ev.append("no words")
    elif not rw: ev.append("loud but no words (laugh / non-verbal?)")
    elif m>=0.6: s+=3; ev.append("echoes Liron: %s"%" ".join(rw))
    elif m>=0.3: s+=1; ev.append("partly echoes Liron (%d%%)"%(m*100))
    if rel<=-18: s+=3; ev.append("%.0f dB under speech"%rel)
    elif rel<=-10: s+=2; ev.append("%.0f dB under speech"%rel)
    elif rel<=-6: s+=1; ev.append("%.0f dB under speech"%rel)
    if r["dur"]<=0.4: s+=1
    if r["own_dist"]>1.5: s+=1; ev.append("far from own speech")
    if rw and m<0.3:
        if all(w in BC for w in rw): s-=3; ev.append("backchannel: %s"%" ".join(rw))
        elif len(rw)>=3 and rel>-6: s-=4; ev.append("real speech: %s"%" ".join(rw))
        elif len(rw)>=1: s-=1.5; ev.append("words: %s"%" ".join(rw))
    if r["own_dist"]<=0.3: s-=2; ev.append("adjacent to own speech")
    if r["why"]=="sustained low level under reference": s+=3; ev.append("gate stuck open %.0fs"%r["dur"])
    if r["peak"]<=-40: s=max(s,6); ev.append("inaudible")
    p=1/(1+math.exp(-(s-2.5)))
    r.update(rel=round(rel,1),words=" ".join(rw) if rw else "",liron=" ".join(lw),lex=round(m,2),score=s,p=round(p,2),evidence="; ".join(ev))
    r["tier"]="remove" if p>=0.85 else "keep" if p<=0.25 else "review"
from collections import Counter
C=Counter(r["tier"] for r in rows); print(C)
rv=[r for r in rows if r["tier"]=="review"]; print("review total %.1fs"%sum(r["dur"] for r in rv))
json.dump(dict(s90=s90,rows=rows),open("scored.json","w"),ensure_ascii=False)
def mmss(t): return "%d:%05.2f"%(t//60,t%60)
with open("overlaps.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["start","end","dur_s","peak_dbfs","vs_speech_db","ncc","richard_words","liron_words_before","lexical_match","p_artifact","tier","evidence"])
    for r in rows:
        if r["tier"]=="n/a": continue
        w.writerow([mmss(r["t0"]),mmss(r["t1"]),"%.2f"%r["dur"],"%.1f"%r["peak"],r["rel"],"%.2f"%r["ncc"],r["words"],r["liron"],r["lex"],r["p"],r["tier"],r["evidence"]])
print("\n-- KEEP examples"); [print(mmss(r["t0"]),r["dur"],r["peak"],r["p"],r["evidence"]) for r in [x for x in rows if x["tier"]=="keep"][:12]]
print("\n-- REVIEW"); [print(mmss(r["t0"]),r["dur"],r["peak"],r["p"],r["evidence"]) for r in rv]
print("\n-- REMOVE with words"); [print(mmss(r["t0"]),r["dur"],r["peak"],r["p"],r["evidence"]) for r in [x for x in rows if x["tier"]=="remove" and x["words"]][:25]]
