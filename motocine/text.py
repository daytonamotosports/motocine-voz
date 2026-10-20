"""Texto de la narración: normalización, tiempos de palabras y anclas.

Un *ancla* es la forma de decir "en el momento en que la voz dice X":

    "pistón"        inicio de la primera vez que se dice "pistón"
    "aceite#2"      inicio de la segunda vez que se dice "aceite"
    "trabar$"       FINAL de la palabra "trabar"
    "Adentro-0.25"  0.25 s antes de "Adentro"   (también +0.3)
    12.5            segundo exacto del video
    "@+1.2"         1.2 s después del inicio de la escena

Las mayúsculas, tildes y signos no importan: "cigüeñal" == "CIGUENAL".
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict

PUNCT_END = (',', '.', ';', ':', '?', '!', '…')


@dataclass
class Word:
    text: str          # como se muestra en subtítulos (con puntuación)
    start: float       # segundos desde el inicio del audio de voz
    end: float

    @property
    def norm(self) -> str:
        return norm(self.text)

    def to_dict(self) -> dict:
        return asdict(self)


def norm(s: str) -> str:
    s = unicodedata.normalize('NFD', str(s).lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9]+', '', s)


def tokens(narration: str) -> list[str]:
    """Palabras de la narración tal como se escriben (sin signos sueltos)."""
    return [t for t in narration.split() if norm(t)]


def estimate_words(toks: list[str], speech_start: float, speech_end: float) -> list[Word]:
    """Reparte el tiempo de habla entre palabras según su largo y las pausas de puntuación.

    Se usa cuando el motor de voz no entrega tiempos por palabra.
    """
    weights, pauses = [], []
    for t in toks:
        weights.append(len(norm(t)) + 1.6)
        pauses.append(4.0 if t.endswith(('.', '?', '!', ':', '…')) else 2.2 if t.endswith((',', ';')) else 0.0)
    total = sum(weights) + sum(pauses[:-1])
    span = max(0.1, speech_end - speech_start)
    k = span / total if total else 0
    out, cur = [], speech_start
    for t, w, p in zip(toks, weights, pauses):
        out.append(Word(t, round(cur, 3), round(cur + w * k, 3)))
        cur += (w + p) * k
    return out


def align(toks: list[str], spoken: list[tuple[str, float, float]], fallback_end: float | None = None) -> list[Word]:
    """Asigna a cada palabra escrita el tiempo de la(s) palabra(s) que dijo la voz.

    `spoken` viene del motor de voz o del transcriptor: [(texto, inicio, fin)].
    Tolera palabras partidas ("cámbialo" -> "cambia" "lo"), números leídos en letras
    y pequeñas diferencias, re-sincronizando con las palabras siguientes.
    """
    out: list[Word] = []
    j = 0
    n = len(spoken)
    for tok in toks:
        tn = norm(tok)
        if j >= n:
            out.append(Word(tok, -1, -1))
            continue
        take = None
        acc = ''
        for k in range(1, 5):
            if j + k > n:
                break
            acc += norm(spoken[j + k - 1][0])
            if acc == tn:
                take = k
                break
            if not tn.startswith(acc):
                break
        if take is None:
            for d in range(1, 4):  # re-sincronizar si la voz insertó palabras (p. ej. números)
                if j + d < n and norm(spoken[j + d][0]) == tn:
                    j += d
                    take = 1
                    break
        if take is None:
            take = 1
        seg = spoken[j:j + take]
        out.append(Word(tok, round(seg[0][1], 3), round(seg[-1][2], 3)))
        j += take
    # rellenar las que quedaron sin tiempo
    missing = [i for i, w in enumerate(out) if w.start < 0]
    if missing:
        last = max([w.end for w in out if w.start >= 0] + [0.0])
        end = fallback_end if fallback_end and fallback_end > last else last + 0.35 * len(missing)
        est = estimate_words([out[i].text for i in missing], last + 0.05, end)
        for i, e in zip(missing, est):
            out[i] = e
    return out


ANCHOR = re.compile(r'^\s*(?P<word>[^\s#$+]+?)(?:#(?P<n>\d+))?(?P<end>\$)?\s*(?P<off>[+-]\s*\d+(?:\.\d+)?)?\s*$')


class AnchorError(ValueError):
    pass


def resolve_anchor(value, words: list[Word], offset: float = 0.0, scene_start: float | None = None) -> float:
    """Convierte un ancla en segundos del video. `offset` = momento en que empieza la voz."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise AnchorError(f'Ancla inválida: {value!r}')
    v = value.strip()
    if v.startswith('@'):
        if scene_start is None:
            raise AnchorError(f'"{value}": "@" solo sirve dentro de una escena')
        rest = v[1:].replace(' ', '') or '+0'
        return scene_start + float(rest)
    m = ANCHOR.match(v)
    if not m:
        raise AnchorError(f'Ancla inválida: "{value}"')
    target = norm(m.group('word'))
    nth = int(m.group('n') or 1)
    hits = [w for w in words if w.norm == target]
    if len(hits) < nth:
        similar = sorted({w.text for w in words if target[:3] and w.norm.startswith(target[:3])})
        found = f'aparece {len(hits)} vez/veces' if hits else 'no aparece en la narración'
        raise AnchorError(f'Ancla "{value}": la palabra "{m.group("word")}" {found}.'
                          + (f' Parecidas: {", ".join(similar)}' if similar else ''))
    w = hits[nth - 1]
    t = w.end if m.group('end') else w.start
    off = float(m.group('off').replace(' ', '')) if m.group('off') else 0.0
    return round(offset + t + off, 3)


def is_time_key(k: str) -> bool:
    """Campos que llevan un tiempo o un ancla: "at", "until", "algo_at" y "algo_until"."""
    return k in ('at', 'until') or k.endswith('_at') or k.endswith('_until')


def resolve_tree(obj, words: list[Word], offset: float, scene_start: float | None = None):
    """Resuelve recursivamente todo campo de tiempo (ver is_time_key)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if is_time_key(k) and v is not None and not isinstance(v, (dict, list)):
                out[k] = resolve_anchor(v, words, offset, scene_start)
            else:
                out[k] = resolve_tree(v, words, offset, scene_start)
        return out
    if isinstance(obj, list):
        return [resolve_tree(v, words, offset, scene_start) for v in obj]
    return obj
