# cgi.py (Python 3.13 임시 호환용 더미)
import urllib.parse as _parse

def parse_header(line):
    """feedparser 호환용 간단한 cgi.parse_header 대체"""
    parts = line.split(";")
    key = parts[0].strip().lower()
    pdict = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.strip().split("=", 1)
            pdict[k.lower()] = v.strip('"')
    return key, pdict
