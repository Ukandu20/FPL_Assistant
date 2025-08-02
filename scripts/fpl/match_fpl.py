#!/usr/bin/env python3
"""
match_fpl_to_fbref.py – link FPL rows to FBref ids (v4+patch).

Key points
----------
✓ Accent-aware & ASCII-folded matching               (Kalajdžić ↔ kalajdzic)
✓ Hyphen / apostrophe first-names handled            (Mihai-Alexandru)
✓ Nickname alias table for short forms               (Edward → Eddie, Daniel → Dani)
✓ Low-value surname words ignored in scoring         (da, dos, van …)
✓ Logs unresolved rows with detailed token info.

Outputs
-------
<out_root>/<season>/players_lookup_enriched.csv|json
<out_root>/_manual_review/unmatched_<season>.json
"""
from __future__ import annotations
import argparse, json, logging, html, re, sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from unidecode import unidecode

# ─────────────────────────  NICKNAMES  ─────────────────────────
NICK: dict[str, list[str]] = {
    "edward":  ["eddie", "ed"],
    "daniel":  ["dani", "danny", "dan"],
    "nicholas":["nick", "niko", "nico"],
    "joshua":  ["josh"],
    "matthew": ["matt"],
    "dominic": ["dom"],
    "frederico": ["fred", "freddie", "freddy"],
    "maximillian": ["max"],
    "alexander": ["alex", "aleks", "sasha"],
    "sasa":    ["saša", "sasha"],
    "emiliano": ["emil", "emi"],
    "Oluwasemilogo": ["semilogo", "semi"],
    "Ademipo": ["mipo", "adeyemi"],
    "Ayotomiwa": ["ayo", "tomi", "tomiwa", "tom"],
    "Jorge": ["jorge", "jorginho"],
    "souza": ["sousa", "Royal"],
    "Vitalii": ["vitaliy", "vitaly", "vitalii"],
    "Vitor": ["vitor", "vitinho", "vitinha"],
    "Joseph": ["joe", "joey"],
    
}

# Low-value tokens in surnames → zero weight
LOW_VALUE = {
    "da","de","do","dos","das",
    "van","von","der","den",
    "di","du","la","le","el","del","della","al","bin"
}

# ──────────────────────  I/O helpers  ─────────────────────────
def jload(p: Path)->dict:
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}

def jdump(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# ──────────────────  canonicalisation  ────────────────────────
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
SUFF  = re.compile(r"_[0-9]+$")

def canon(s:str, *, fold=False)->str:
    s = html.unescape(s)
    s = SUFF.sub("", s).replace("_"," ").replace("-"," ")
    if fold: s = unidecode(s)
    s = PUNCT.sub(" ", s).lower()
    return " ".join(s.split())

def variants(tok:str)->List[str]:
    a, b = canon(tok), canon(tok, fold=True)
    return [a] if a==b else [a,b]

def expand_first_token(raw:str)->List[str]:
    """full+folded + split parts if hyphen / apostrophe + nicknames."""
    base = variants(raw)
    if "-" in raw or "'" in raw:
        for part in re.split(r"[-']", raw):
            base.extend(variants(part))
    base.extend( sum((variants(n) for n in NICK.get(canon(raw), [])), []) )
    return sorted(set(base))

def expand_optional_token(tok:str)->List[str]:
    """optional first-name tokens ⇒ variants + nicknames."""
    base = variants(tok)
    base.extend(sum((variants(n) for n in NICK.get(canon(tok), [])), []))
    return sorted(set(base))

# ────────────────────  lookup table  ──────────────────────────
def build_lookup(fp:Path)->Dict[str,Tuple[str,str]]:
    raw=jload(fp); out={}
    for key_acc,pid in raw.items():
        out[key_acc]=(key_acc,pid)
        out[canon(key_acc,fold=True)]=(key_acc,pid)
    return out

# ────────────────────  matching logic  ────────────────────────
def match_row(fn_raw:str, sn_raw:str,
              lu:Dict[str,Tuple[str,str]]
             )->Tuple[str|None,str|None,dict]:
    fn_tokens=[canon(t) for t in fn_raw.split() if canon(t)]
    sn_tokens=[canon(t) for t in sn_raw.split() if canon(t)]
    if not fn_tokens:
        return None,None,{"reason":"empty-first"}

    first_mand = fn_tokens[0]
    first_opt  = fn_tokens[1:]

    # Step-1: mandatory first token (with nick / split) must appear
    cand = {k:v for k,v in lu.items()
            if any(var in k for var in expand_first_token(first_mand))}
    if not cand:
        return None,None,{"reason":"no-first-token"}

    # Step-2: prefer keys that hit at least one second-name token
    cand2 = {k:v for k,v in cand.items()
             if any(any(var in k for var in variants(t)) for t in sn_tokens)} or cand
    if len(cand2)==1:
        _,(fb,pid)=next(iter(cand2.items())); return fb,pid,{}

    # Step-3: overlap score, ignoring low-value tokens
    scores:Dict[str,float]={}
    for k in cand2:
        sec_hits=sum(any(var in k for var in variants(t))
                     for t in sn_tokens if t not in LOW_VALUE)
        opt_hits=sum(any(var in k for var in expand_optional_token(t))
                     for t in first_opt)
        denom = (sum(1 for t in sn_tokens if t not in LOW_VALUE)
                 + max(1,len(first_opt)))
        scores[k]=(sec_hits+opt_hits)/denom

    best=max(scores.values())
    best_keys=[k for k,v in scores.items() if v==best]

    ids={lu[k][1] for k in best_keys}
    if len(ids)==1:
        fb,pid=lu[best_keys[0]]
        return fb,pid,{}

    return None,None,{"candidates":best_keys,"best_score":best,"ids":list(ids)}

# ─────────────────────  per-season  ───────────────────────────
def process_season(inf:Path, lu:dict, outf:Path, revf:Path):
    df = pd.read_json(inf) if inf.suffix==".json" else pd.read_csv(inf)
    names,ids,review=[],[],[]
    for _, r in df.iterrows():
        fb, pid, log = match_row(str(r['first_name']),
                                str(r['second_name']), lu)

        # ↓↓  compute tokens first (no walrus)  ↓↓
        first_tok  = [canon(t) for t in r['first_name'].split()]
        second_tok = [canon(t) for t in r['second_name'].split()]

        if fb:
            names.append(fb)
            ids.append(pid)
        else:
            names.append(f"{r['first_name']} {r['second_name']}")
            ids.append(None)
            review.append({
                "first_name":   r['first_name'],
                "second_name":  r['second_name'],
                "first_tokens": first_tok,
                "second_tokens": second_tok,
                **log
            })
    df["name"],df["player_id"]=names,ids
    outf.parent.mkdir(parents=True, exist_ok=True)
    if outf.suffix==".json": jdump(outf, df.to_dict(orient="records"))
    else:                     df.to_csv(outf,index=False)
    if review: jdump(revf,review); logging.info("✖︎ %d unmatched → %s",len(review),revf)
    else:      revf.unlink(missing_ok=True); logging.info("✔︎ all players matched")

# ──────────────────────────  CLI  ─────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--players-root",type=Path,required=True)
    ap.add_argument("--lookup-file", type=Path,required=True)
    ap.add_argument("--out-root",    type=Path)
    ap.add_argument("--format",choices=("csv","json"))
    ap.add_argument("--log-level",default="INFO")
    args=ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s: %(message)s")
    lu=build_lookup(args.lookup_file)
    root_out=args.out_root or args.players_root
    processed=0
    for sd in sorted(args.players_root.iterdir()):
        if not sd.is_dir(): continue
        in_file = next((sd/f for f in ("players_lookup.csv","players_lookup.json")
                        if (sd/f).exists()), None)
        if not in_file: continue
        ext=args.format or in_file.suffix
        outf   = root_out/sd.name/f"players_lookup_enriched{ext}"
        revf   = root_out/"_manual_review"/f"unmatched_{sd.name}.json"
        logging.info("Season %s …", sd.name)
        process_season(in_file, lu, outf, revf); processed+=1
    if not processed:
        logging.error("No seasons processed – check paths."); sys.exit(1)
    logging.info("Finished %d season(s) ✅", processed)

if __name__ == "__main__":
    main()
