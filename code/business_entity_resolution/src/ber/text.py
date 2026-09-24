"""Text normalization primitives.

Hand-written rules here are *formatting only* (Unicode/case/punctuation cleanup,
placeholder removal, leading zeros, splitting "X dba Y" wrappers and compact
domain-style names). Every value mapping (native-script words, state/street aliases)
is learned from the training portion by `ber.learn_maps` and passed in as `maps`.

Each rule, with its basis in the supplied files:
* NFKC + unidecode: accents and Unicode variants ("Café" vs "Cafe").
* "&" -> "and": README noise pattern "& vs. and".
* dots/apostrophes/# removed inside words: "L.L.C." == "LLC", "Example's", "#831".
* split on other punctuation incl. "-" and "/": "Tri-State", "40/3", "C/O".
* numbers lose leading zeros: "0040" vs "40" (seen in train pairs).
* S3 placeholders "NULL"/"<NULL>" dropped.
* "X dba/aka/fka/t/a/... Y": the S1 name is the part after the marker (20k/20k
  sampled train pairs), so Y is the main name and X goes to the alt field.
* "Name | www.site.com": the part before "|" is the main name.
* domain-style names ("examplebusiness.com", "#examplebusiness"): TLD/"#" stripped
  and the compact string segmented with a unigram model of the same split's S1 name
  vocabulary (label-free).
"""
import math
import re
import unicodedata
from functools import lru_cache

from unidecode import unidecode

INDIC = re.compile(r"[ऀ-෿]")
WORD_SPLIT = re.compile(r"[\s,;:()\[\]{}\"|+*<>=_~^%$!?/\\\-]+")
IN_WORD_DROP = re.compile(r"[.'’`#@]")
NON_ALNUM = re.compile(r"[^a-z0-9]")
DBA = re.compile(
    r"\s+(?:d\.?\s?b\.?\s?a\.?|t/a|formerly known as|f/k/a|fka|a/k/a|aka|trading as|doing business as)\s+",
    re.IGNORECASE,
)
DOMAIN_WORD = re.compile(r"^(?:www\.)?([^\s]+?)\.(?:com|net|org)$", re.IGNORECASE)
PLACEHOLDER = re.compile(r"<\s*null\s*>|\bnull\b", re.IGNORECASE)
LATIN_LETTER = re.compile(r"[A-Za-z]")


@lru_cache(maxsize=2_000_000)
def ud(word: str) -> str:
    """unidecode + lowercase + keep [a-z0-9] only (one output token per input word)."""
    return NON_ALNUM.sub("", unidecode(word).lower())


def _norm_num(tok: str) -> str:
    return str(int(tok)) if tok.isdigit() and len(tok) < 19 else tok


def raw_words(text: str) -> list:
    """Split on separators; in-word dots/apostrophes/# are kept (domain detection)."""
    text = unicodedata.normalize("NFKC", text).replace("&", " and ")
    return [w for w in WORD_SPLIT.split(text) if w]


def words(text: str) -> list:
    """Split into cleaned raw words (native script untouched)."""
    return [w for w in (IN_WORD_DROP.sub("", w) for w in raw_words(text)) if w]


def word_key(w: str) -> str:
    """Normalized key of one raw word (same for learning and applying maps)."""
    return _norm_num(ud(w))


class Segmenter:
    """Unigram word segmentation of compact domain-style names."""

    def __init__(self, counts: dict, max_len: int = 24):
        total = sum(counts.values()) or 1
        self.cost = {w: -math.log(c / total) for w, c in counts.items() if len(w) > 1 or w.isdigit()}
        self.unk = -math.log(0.5 / total) + 5.0
        self.max_len = max_len

    def __call__(self, s: str) -> list:
        n = len(s)
        if n == 0 or n > 80:
            return [s] if s else []
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_len), i):
                piece = s[j:i]
                c = self.cost.get(piece, self.unk * len(piece) if i - j == 1 else math.inf)
                if best[j] + c < best[i]:
                    best[i], back[i] = best[j] + c, j
        out, i = [], n
        while i > 0:
            out.append(s[back[i]:i])
            i = back[i]
        return out[::-1]


def norm_name(raw: str, native_map: dict = None, segmenter: Segmenter = None) -> dict:
    """Normalize a business name. Returns main tokens plus flags."""
    raw = raw or ""
    s = unicodedata.normalize("NFKC", raw).strip()
    alt = []
    if "|" in s:  # "Name | www.site.com"
        s, rest = s.split("|", 1)
        alt.append(rest)
    parts = DBA.split(s, maxsplit=1)
    is_dba = len(parts) == 2
    if is_dba:
        alt.append(parts[0])
        s = parts[1]
    native = bool(INDIC.search(s))
    toks, n_native, n_mapped, is_domain = [], 0, 0, False
    for w in raw_words(s):
        m = DOMAIN_WORD.match(w) if "." in w else None
        hashtag = w.startswith("#") and len(ud(w)) >= 6 and not ud(w).isdigit()
        if (m or hashtag) and segmenter is not None:
            toks.extend(segmenter(ud(m.group(1) if m else w)))
            is_domain = True
            continue
        w = IN_WORD_DROP.sub("", w)
        k = word_key(w)
        if INDIC.search(w):
            n_native += 1
            if native_map and k in native_map:
                k = native_map[k]
                n_mapped += 1
        if k:
            toks.append(k)
    letters = LATIN_LETTER.findall(raw)
    return {
        "n_tok": toks,
        "n_alt": " ".join(t for a in alt for t in (word_key(w) for w in words(a)) if t) or None,
        "n_native": native,
        "n_native_mapped_frac": (n_mapped / n_native) if n_native else None,
        "n_allcaps": bool(letters) and raw.upper() == raw,
        "n_accent": any(ord(ch) > 127 and not INDIC.match(ch) for ch in raw),
        "n_domain": is_domain,
        "n_dba": is_dba,
    }


def addr_components(raw: str) -> list:
    """Comma-separated components with placeholders removed (raw text)."""
    if not raw:
        return []
    s = PLACEHOLDER.sub(" ", unicodedata.normalize("NFKC", raw))
    return [c.strip() for c in s.split(",") if c.strip()]


def comp_key(comp: str) -> str:
    return " ".join(k for k in (word_key(w) for w in words(comp)) if k)


def norm_addr(raw: str, comp_map: dict = None, tok_map: dict = None) -> dict:
    """Normalize an address: component-level aliases, then token-level aliases."""
    comps = addr_components(raw)
    keys = []
    native = False
    for c in comps:
        native = native or bool(INDIC.search(c))
        k = comp_key(c)
        if comp_map and k in comp_map:
            k = comp_map[k]
        if k:
            keys.append(k)
    toks = [t for k in keys for t in k.split(" ")]
    if tok_map:
        toks = [tok_map.get(t, t) for t in toks]
        toks = [t for tt in toks for t in tt.split(" ")]
    nums = [t for t in toks if t.isdigit()]
    return {
        "a_comp": keys,
        "a_tok": toks,
        "a_nums": nums,
        "a_null": len(comps) == 0,
        "a_native": native,
        "a_ncomma": max(0, len(comps) - 1),
    }


def is_abbrev(a: str, b: str) -> bool:
    """a (>=2 chars, shorter than b, same first letter) is a prefix or subsequence of b.

    Weak evidence only (PLAN B1): "ave"/"avenue", "blvd"/"boulevard", "tn"/"tennessee".
    """
    if len(a) < 2 or len(a) >= len(b) or a[0] != b[0]:
        return False
    if b.startswith(a):
        return True
    it = iter(b)
    return all(ch in it for ch in a)
