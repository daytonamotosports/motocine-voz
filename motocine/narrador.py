"""Narrador: decide CÓMO se dice cada frase, como lo haría un locutor de redes, sin que nadie lo escriba a mano.

Lee el guion frase por frase y le asigna un papel; cada papel tiene su ritmo, su tono y su pausa:

  gancho      la primera frase: un poco más pausada y más arriba de tono (que se entienda TODO desde el inicio)
  pregunta    termina en "?": más lenta y un poco más alta, con respiro después (deja pensar al que mira)
  intriga     "nadie", "nunca", "secreto", "el tercero", "pero"...: más baja y lenta, con silencio ANTES (suspenso)
  numero      "Uno: ...", "Dos: ...": el número se separa con una pausa corta dentro de la misma frase
              ("Uno… llantas bajas"); dicho solo, la voz lo lee en inglés ("Tres." → "Trash")
  titular     "Primero, la cadena:" (antes de los dos puntos): se anuncia más pausado y fuerte, respira y explica
  remate      frases cortas que cierran una idea ("Un motor nuevo, muchísimo."): golpe, pausa después
  cta         el cierre (comenta, síguenos): más ágil y arriba, con energía
  normal      el resto, con una variación mínima de tono y ritmo para que no suene a metrónomo

Todo es local y barato: el tono se cambia con rubberband (ffmpeg) conservando el timbre, y cada frase ya
hecha se guarda en caché (cambiar una frase del guion no vuelve a sintetizar las demás).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

INTRIGA = ('nadie', 'nunca', 'jamas', 'secreto', 'el tercero', 'la tercera', 'el ultimo', 'la ultima', 'error', 'peor',
           'cuidado', 'ojo', 'pero', 'sin embargo', 'en realidad', 'la verdad', 'resulta que', 'lo que pasa', 'casi nadie')
CTA = ('comenta', 'siguenos', 'sigueme', 'escribenos', 'guarda', 'comparte', 'agenda', 'pidelo')
NUMERO = re.compile(r'^\s*(uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|primero|segundo|tercero|'
                    r'\d+)\s*[:.,]\s+', re.I)


def _plain(s: str) -> str:
    s = unicodedata.normalize('NFD', s.lower())
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn')


def _jit(text: str, k: int = 0) -> float:
    """Variación fija (-0.5..0.5) por frase: el mismo guion suena igual cada vez."""
    return int(hashlib.md5(f'{k}{text}'.encode()).hexdigest()[:4], 16) / 65535 - .5


def role(text: str, i: int, n: int) -> str:
    p = _plain(text)
    if text.rstrip().endswith(',') and i < n - 1:
        return 'tramo'
    if text.rstrip().endswith(':') and i < n - 1:
        return 'titular'
    if i == n - 1 and any(w in p for w in CTA):
        return 'cta'
    if i == 0:
        return 'gancho'
    if NUMERO.match(text):
        return 'numero'
    if any(re.search(r'\b' + w + r'\b', p) for w in INTRIGA):
        return 'intriga'
    if text.rstrip().endswith('?'):
        return 'pregunta'
    if len(text.split()) <= 5:
        return 'remate'
    return 'normal'


#               ritmo  tono (semitonos)  pausa antes  pausa después
# pausas medidas contra locutores de redes (20-oct, "lo dice muy seguido, no da los tiempos"): después del gancho
# y de la intriga se deja un silencio de verdad para que el que mira se quede esperando lo que sigue
STYLE = {'gancho':   (.95,  .5, 0.00, .55),
         'pregunta': (.94,  .4, 0.00, .55),
         'intriga':  (.88, -.6, 0.30, .75),
         'numero':   (1.0,  .2, 0.30, .30),
         'remate':   (.92, -.3, 0.10, .45),
         'cta':      (1.02, .4, 0.25, .10),
         'normal':   (1.0,  0., 0.00, .30),
         'tramo':    (1.0,  .1, 0.00, .22),    # primera mitad de una frase partida antes del suspenso
         'titular':  (.92,  .35, 0.15, .30)}   # "Primero, la cadena:" — se anuncia, se respira y luego se explica
# fuerza de la voz por papel (%): lo que engancha va más fuerte, el suspenso un poco más bajo (como al oído)
VOLUME = {'gancho': 8, 'titular': 7, 'intriga': -4, 'cta': 6, 'remate': 4}


def beats(chunks: list[list[str]]) -> list[list[str]]:
    """Parte una frase en la coma que va justo antes de la intriga: "Hay tres culpables, | y el tercero casi
    nadie lo revisa." Cada mitad se dice aparte, con un silencio entre las dos (suspenso)."""
    out = []
    for ch in chunks:
        cut = None
        for i in range(len(ch) - 1):
            if ch[i].endswith(',') and i >= 1:
                rest = _plain(' '.join(ch[i + 1:i + 5]))
                rest = rest[2:] if rest.startswith('y ') else rest
                if any(rest.startswith(w) or (' ' + w) in ' ' + rest[:24] for w in INTRIGA):
                    cut = i + 1
        if cut and len(ch) - cut >= 2:
            parts = [ch[:cut], ch[cut:]]
        else:
            parts = [ch]
        for pt in parts:                             # "Primero, la cadena: | si cuelga mucho…" (el titular se anuncia)
            k = next((i + 1 for i, w in enumerate(pt[:-2]) if w.endswith(':') and i >= 1), None)
            out += [pt[:k], pt[k:]] if k else [pt]
    return out


def plan(sentences: list[str]) -> list[dict]:
    """[{text, say, role, speed_k, pitch, pause_before, pause_after}] para cada frase del guion."""
    out, n = [], len(sentences)
    for i, s in enumerate(sentences):
        r = role(s, i, n)
        sp, pt, pb, pa = STYLE[r]
        say = s
        if r == 'numero':                                   # "Uno: llantas" → "Uno… llantas" (pausa dentro)
            say = NUMERO.sub(lambda m: m.group(1) + '… ', s, count=1)
        j = _jit(s)
        out.append({'text': s, 'say': say, 'role': r, 'speed_k': round(sp * (1 + .04 * j), 4), 'volume': VOLUME.get(r, 0),
                    'pitch': round(pt + .3 * _jit(s, 1), 3), 'pause_before': pb, 'pause_after': round(pa + .04 * j, 3)})
    return out


def pitch_shift(x, sr: int, semitones: float):
    """Cambia el tono sin cambiar la duración ni el timbre (rubberband vía ffmpeg). Sin cambios si es mínimo."""
    import subprocess

    import numpy as np
    if abs(semitones) < .05 or not len(x):
        return x
    f = 2 ** (semitones / 12)
    try:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(sr), '-ac', '1', '-i', '-',
                            '-af', f'rubberband=pitch={f:.5f}:formant=preserved:pitchq=quality:transients=smooth',
                            '-f', 'f32le', '-'], input=np.asarray(x, np.float32).tobytes(), capture_output=True, check=True)
        y = np.frombuffer(r.stdout, np.float32)
        return y if len(y) > .9 * len(x) else x
    except Exception:
        return x


def breath(sr: int, dur: float = .2, level: float = .006, seed: int = 0):
    """Respiración suave antes de una frase tras una pausa larga: ruido filtrado con forma de inhalación."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    w = rng.standard_normal(n).astype(np.float32)
    k = np.ones(12, np.float32) / 12                       # quita lo más agudo (sin siseo)
    w = np.convolve(w, k, 'same') - np.convolve(w, np.ones(200, np.float32) / 200, 'same')   # banda media
    env = np.sin(np.linspace(0, np.pi, n)) ** 1.5
    return (w * env * level / (np.abs(w).max() + 1e-6) * 4).astype(np.float32)
