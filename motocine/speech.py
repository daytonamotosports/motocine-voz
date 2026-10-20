"""Pronunciación y vigilante de la voz.

pronounce(text, extra)  Convierte el texto del guion en lo que la voz debe DECIR: números en letras ("250" →
                        "doscientos cincuenta", "$25.000" → "veinticinco mil pesos"), referencias de motos
                        ("NS 200" → "ene ese doscientos", "N250" → "ene doscientos cincuenta"), siglas y palabras
                        en inglés del taller ("o-ring", "km"...). Los subtítulos y anclas siguen usando el guion
                        original; solo cambia lo que se pronuncia. Cada proyecto puede agregar las suyas en
                        voice.pronounce: {"Duke": "Diuk"}.

Checker                 Vigilante: escucha cada frase con un reconocedor de voz local (Whisper vía sherpa-onnx,
                        gratis y sin internet; ~640 MB que se bajan una vez de GitHub) y compara lo que entendió
                        con lo que debía decir. Marca las palabras que no se entienden o quedan cortadas.
"""
from __future__ import annotations

import difflib
import os
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------- números en letras (español)
_U = ['cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez', 'once', 'doce', 'trece',
      'catorce', 'quince', 'dieciséis', 'diecisiete', 'dieciocho', 'diecinueve', 'veinte', 'veintiuno', 'veintidós',
      'veintitrés', 'veinticuatro', 'veinticinco', 'veintiséis', 'veintisiete', 'veintiocho', 'veintinueve']
_D = ['', '', '', 'treinta', 'cuarenta', 'cincuenta', 'sesenta', 'setenta', 'ochenta', 'noventa']
_C = ['', 'ciento', 'doscientos', 'trescientos', 'cuatrocientos', 'quinientos', 'seiscientos', 'setecientos',
      'ochocientos', 'novecientos']


def num_words(n: int, apocope: bool = False) -> str:
    """0..999 999 999 en letras. apocope=True: "un" en vez de "uno" (antes de "mil", "pesos"...)."""
    if n < 0:
        return 'menos ' + num_words(-n, apocope)
    if n < 30:
        w = _U[n]
        if apocope:
            w = {'uno': 'un', 'veintiuno': 'veintiún'}.get(w, w)
        return w
    if n < 100:
        d, u = divmod(n, 10)
        return _D[d] + ('' if u == 0 else ' y ' + num_words(u, apocope))
    if n < 1000:
        c, r = divmod(n, 100)
        if n == 100:
            return 'cien'
        return _C[c] + ('' if r == 0 else ' ' + num_words(r, apocope))
    if n < 1_000_000:
        m, r = divmod(n, 1000)
        head = 'mil' if m == 1 else num_words(m, True) + ' mil'
        return head + ('' if r == 0 else ' ' + num_words(r, apocope))
    mm, r = divmod(n, 1_000_000)
    head = 'un millón' if mm == 1 else num_words(mm, True) + ' millones'
    return head + ('' if r == 0 else ' ' + num_words(r, apocope))


# ---------------------------------------------------------------- pronunciación
TALLER = {                       # palabras del taller que la voz lee mal tal cual
    'o-ring': 'o ring', 'o-rings': 'o rings', 'km': 'kilómetros', 'cc': 'centímetros cúbicos', 'rpm': 'revoluciones',
    'abs': 'a be ese', 'cbs': 'ce be ese', 'led': 'led', 'dot4': 'dot cuatro', 'dot 4': 'dot cuatro',
    'duke': 'diuk', 'w': 'w', 'ok': 'okey',
}


def pronounce(text: str, extra: dict | None = None) -> str:
    s = ' ' + text + ' '
    for k, v in sorted({**TALLER, **{k.lower(): v for k, v in (extra or {}).items()}}.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(r'(?<![\wáéíóúñ-])' + re.escape(k) + r'(?![\wáéíóúñ-])', v, s, flags=re.I)
    # referencias de motos: NS 200, N250, RS 200, FZ 2.0 ...
    s = re.sub(r'\bNS\s?(\d+)', lambda m: 'ene ese ' + num_words(int(m.group(1))), s)
    s = re.sub(r'\bRS\s?(\d+)', lambda m: 'erre ese ' + num_words(int(m.group(1))), s)
    s = re.sub(r'\b([A-Z])\s?(\d{2,4})\b', lambda m: _letter(m.group(1)) + ' ' + num_words(int(m.group(2))), s)
    # viscosidad de aceites: 20W-50 → veinte w cincuenta
    s = re.sub(r'\b(\d+)\s?[wW]\s?-?\s?(\d+)\b', lambda m: num_words(int(m.group(1))) + ' uve ' + num_words(int(m.group(2))), s)
    # dinero: $25.000 → veinticinco mil pesos
    s = re.sub(r'\$\s?(\d{1,3}(?:[.,]\d{3})+|\d+)', lambda m: num_words(int(re.sub(r'[.,]', '', m.group(1))), True) + ' pesos', s)
    # miles con punto (25.000) y números sueltos
    s = re.sub(r'\b\d{1,3}(?:\.\d{3})+\b', lambda m: num_words(int(m.group(0).replace('.', ''))), s)
    s = re.sub(r'\b\d+\b', lambda m: num_words(int(m.group(0))), s)
    # palabras en MAYÚSCULAS (no siglas cortas): la voz las deletrea; se pasan a minúscula
    s = re.sub(r'\b[A-ZÁÉÍÓÚÑ]{4,}\b', lambda m: m.group(0).lower(), s)
    return re.sub(r'\s+', ' ', s).strip()


def _letter(c: str) -> str:
    return {'N': 'ene', 'S': 'ese', 'R': 'erre', 'F': 'efe', 'M': 'eme', 'L': 'ele', 'X': 'equis', 'Z': 'zeta',
            'Y': 'ye', 'H': 'hache', 'G': 'ge', 'J': 'jota', 'K': 'ka', 'Q': 'cu', 'V': 'uve', 'W': 'uve doble',
            'B': 'be', 'C': 'ce', 'D': 'de', 'T': 'te', 'P': 'pe'}.get(c.upper(), c)


# ---------------------------------------------------------------- vigilante (reconocimiento de voz)
WHISPER = {'name': 'sherpa-onnx-whisper-small',
           'url': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2'}


def _norm_words(text: str) -> list[str]:
    t = unicodedata.normalize('NFD', pronounce(text).lower())
    t = ''.join(c for c in t if unicodedata.category(c) != 'Mn')
    return re.findall(r"[a-zñ0-9]+", t)


def _whisper_dir(download: bool = True) -> Path | None:
    base = Path(os.getenv('MOTOCINE_MODELS', Path.home() / '.cache' / 'motocine' / 'models'))
    d = base / WHISPER['name']
    if (d / 'small-encoder.int8.onnx').exists():
        return d
    if not download:
        return None
    import shutil
    import tarfile
    import tempfile
    import urllib.request
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.tar.bz2', dir=base, delete=False) as tmp:
        with urllib.request.urlopen(WHISPER['url'].format(name=WHISPER['name']), timeout=1800) as r:
            shutil.copyfileobj(r, tmp)
    with tarfile.open(tmp.name) as tf:
        tf.extractall(base, filter='data')
    Path(tmp.name).unlink()
    return d


class _quiet:
    """Silencia los mensajes internos del reconocedor (escribe directo a la salida de errores en cada frase)."""
    def __enter__(self):
        import sys
        sys.stderr.flush()
        self.fd = os.dup(2)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 2)
        os.close(null)

    def __exit__(self, *a):
        os.dup2(self.fd, 2)
        os.close(self.fd)


class Checker:
    """Escucha un pedazo de audio y dice qué tan bien se entiende frente al texto esperado."""

    def __init__(self, download: bool = True):
        import sherpa_onnx
        d = _whisper_dir(download)
        if d is None:
            raise FileNotFoundError('modelo Whisper no disponible')
        with _quiet():
            self.rec = self._load(sherpa_onnx, d)

    @staticmethod
    def _load(sherpa_onnx, d):
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(d / 'small-encoder.int8.onnx'), decoder=str(d / 'small-decoder.int8.onnx'),
            tokens=str(d / 'small-tokens.txt'), language='es', task='transcribe', num_threads=4)

    def hear(self, samples, sr: int) -> str:
        import numpy as np
        with _quiet():
            return self._hear(np, samples, sr)

    def _hear(self, np, samples, sr):
        st = self.rec.create_stream()
        pad = np.zeros(int(.25 * sr), np.float32)                    # un poco de silencio a los lados ayuda
        st.accept_waveform(sr, np.concatenate([pad, np.asarray(samples, np.float32), pad]))
        self.rec.decode_stream(st)
        return st.result.text.strip()

    # manías del reconocedor al ESCRIBIR una palabra que la voz dice bien (no son errores de pronunciación)
    SAME = {'bugilla': 'bujia', 'bugia': 'bujia', 'bujilla': 'bujia'}

    def score(self, expected: str, heard: str) -> tuple[float, list[str]]:
        """(0..1 parecido, palabras esperadas que no se oyeron)."""
        a, b = _norm_words(expected), [self.SAME.get(w, w) for w in _norm_words(heard)]
        if not a:
            return 1.0, []
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        missing = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op in ('replace', 'delete'):
                missing += a[i1:i2]
        hit = sum(m.size for m in sm.get_matching_blocks())
        return hit / len(a), missing
