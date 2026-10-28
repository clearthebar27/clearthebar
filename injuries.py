import json, urllib.request, os, time

URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries'

def _cache_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'injuries_cache.json')

def fetch(max_age=1800):
    fp = _cache_path()
    try:
        if os.path.exists(fp) and (time.time() - os.path.getmtime(fp)) < max_age:
            with open(fp, encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    out = {}
    for t in d.get('injuries', []):
        for inj in t.get('injuries', []):
            ath = inj.get('athlete') or {}
            nm = ath.get('displayName') or ''
            st = (inj.get('status') or '').strip()
            if nm:
                out[nm] = st
    try:
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(out, f)
    except Exception:
        pass
    return out