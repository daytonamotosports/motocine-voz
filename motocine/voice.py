"""Narración: genera (o carga) la voz y obtiene el tiempo de cada palabra.

Motores (campo voice.engine del proyecto):
  edge    Voces neuronales de Microsoft Edge. Gratis, sin clave, requiere internet.
          Entrega el tiempo exacto de cada palabra (WordBoundary).
  file    Tu propia grabación (voice.file). Tiempos con faster-whisper si está
          instalado; si no, se estiman.
  piper   Voz local sin internet (https://github.com/rhasspy/piper). Requiere el
          programa `piper` y un modelo .onnx (voice.model). Tiempos estimados o whisper.
  kokoro  POR DEFECTO. Voz neuronal local, gratis y SIN internet (Kokoro vía sherpa-onnx). Voces de
          hombre em_santa (la elegida, velocidad 1.25) y em_alex, mujer ef_dora. El modelo (~130 MB) se baja una
          vez de GitHub. Tiempos: exactos por frase, estimados por palabra dentro de la frase.
  silent  Sin voz: tiempos estimados. Sirve para probar animaciones sin internet.

El resultado se guarda en build/voice/ con una huella del texto y la voz, así que
volver a renderizar no vuelve a llamar al servicio de voz.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import json
import re
import shutil
import subprocess
from pathlib import Path

from .text import Word, align, estimate_words, tokens

VOICE_VERSION = 'v12'          # subir cuando cambie cómo se arma la voz (invalida la caché)
DEFAULT_VOICE = {'engine': 'edge', 'voice': 'es-MX-JorgeNeural', 'speed': 1.15, 'rate': '+0%', 'pitch': '+0Hz', 'start': 0.4}


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def speech_bounds(path: Path, noise_db: int = -38) -> tuple[float, float]:
    """Inicio y fin del habla (ignora silencio al principio y al final)."""
    dur = ffprobe_duration(path)
    r = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(path), '-af', f'silencedetect=noise={noise_db}dB:d=0.12', '-f', 'null', '-'],
                       capture_output=True, text=True)
    starts = [float(v) for v in re.findall(r'silence_start: ([\d.]+)', r.stderr)]
    ends = [float(v) for v in re.findall(r'silence_end: ([\d.]+)', r.stderr)]
    s, e = 0.0, dur
    if starts and starts[0] <= 0.05 and ends:
        s = ends[0]
    if starts and (not ends or starts[-1] > ends[-1]):
        e = starts[-1]
    return s, max(e, s + 0.2)


# modelo completo (no el comprimido int8: sonaba más robótico y se comía el inicio de las frases, 20-oct)
KOKORO = {'name': 'kokoro-multi-lang-v1_0', 'model': 'model.onnx', 'voices': {'ef_dora': 28, 'em_alex': 29, 'em_santa': 53},
          'url': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{name}.tar.bz2'}


def _kokoro_dir() -> Path:
    import os
    base = Path(os.getenv('MOTOCINE_MODELS', Path.home() / '.cache' / 'motocine' / 'models'))
    d = base / KOKORO['name']
    if not (d / KOKORO['model']).exists():
        import tarfile
        import tempfile
        import urllib.request
        base.mkdir(parents=True, exist_ok=True)
        url = KOKORO['url'].format(name=KOKORO['name'])
        with tempfile.NamedTemporaryFile(suffix='.tar.bz2', dir=base, delete=False) as tmp:
            with urllib.request.urlopen(url, timeout=600) as r:
                shutil.copyfileobj(r, tmp)
        with tarfile.open(tmp.name) as tf:
            tf.extractall(base, filter='data')
        Path(tmp.name).unlink()
    return d


def _trim(x, sr, with_start: bool = False, pad0: float = .12, pad1: float = .2):
    """Quita el silencio de los extremos SIN comerse los sonidos suaves (la "s" inicial, finales que se apagan):
    envolvente suavizada con umbral bajo, márgenes amplios y fundidos cortos para que no haya clics."""
    import numpy as np
    if not len(x):
        return (x, 0) if with_start else x
    win = max(1, int(.01 * sr))
    env = np.sqrt(np.convolve(x * x, np.ones(win) / win, 'same'))
    on = env > max(.0035, env.max() * .012)
    if not on.any():
        return (x, 0) if with_start else x
    i0, i1 = int(np.argmax(on)), len(x) - int(np.argmax(on[::-1]))
    a0 = max(0, i0 - int(pad0 * sr))
    y = x[a0:min(len(x), i1 + int(pad1 * sr))].copy()
    f = min(len(y) // 4, int(.008 * sr))
    if f > 1:
        y[:f] *= np.linspace(0, 1, f); y[-f:] *= np.linspace(1, 0, f)
    return (y, a0) if with_start else y


def _kokoro(toks: list[str], cfg: dict, out: Path, log=print):
    """Sintetiza oración por oración (las comas quedan dentro: entonación natural y sin cortes), dice lo que el
    guion quiere decir (números, referencias de motos, ver speech.pronounce) y, si hay vigilante, escucha cada
    oración y la repite con otra variante cuando no se entiende bien."""
    import wave

    import numpy as np
    import sherpa_onnx

    from .speech import Checker, pronounce
    d = _kokoro_dir()
    kc = sherpa_onnx.OfflineTtsKokoroModelConfig(model=str(d / KOKORO['model']), voices=str(d / 'voices.bin'),
                                                 tokens=str(d / 'tokens.txt'), data_dir=str(d / 'espeak-ng-data'),
                                                 dict_dir=str(d / 'dict'), lexicon=str(d / 'lexicon-us-en.txt'),
                                                 lang=cfg.get('acento', 'es-419'))   # 'es' = España ("cinco" con zeta)
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(kokoro=kc, num_threads=4)))
    v = cfg.get('voice') if cfg.get('voice') in KOKORO['voices'] else 'em_alex'   # una voz de Edge en la config no aplica aquí
    sid = KOKORO['voices'][v]
    base_speed = float(cfg.get('speed', 1.0))
    # frases: se corta en . ? ! … (los ":" quedan dentro: "Uno: llantas" lo maneja el narrador)
    chunks, cur = [], []
    for t in toks:
        cur.append(t)
        if t.endswith(('.', '?', '!', '…')):
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    merged = []                     # una palabra sola ("Tres.") se lee en inglés ("Trash"): va con la siguiente
    for ch in chunks:
        if merged and len(merged[-1]) == 1:
            merged[-1] = merged[-1] + ch
        else:
            merged.append(ch)
    chunks = merged
    from . import narrador
    if cfg.get('narrador', True) is not False:
        chunks = narrador.beats(chunks)
    directed = cfg.get('narrador', True) is not False
    plan = narrador.plan([' '.join(ch) for ch in chunks]) if directed else [
        {'text': ' '.join(ch), 'say': ' '.join(ch), 'role': 'normal', 'speed_k': 1.0, 'pitch': 0, 'pause_before': 0,
         'pause_after': .16} for ch in chunks]
    checker = None
    if cfg.get('check', True) is not False:
        try:
            checker = Checker(download=cfg.get('check') == 'download' or bool(os.getenv('MOTOCINE_VOICE_CHECK')))
        except Exception:
            checker = None
    extra = cfg.get('pronounce') or {}
    cache = Path(os.getenv('MOTOCINE_MODELS', Path.home() / '.cache' / 'motocine' / 'models')).parent / 'frases'
    cache.mkdir(parents=True, exist_ok=True)
    audio, words, report, pos, sr = [], [], [], 0.0, 24000
    for ch, pl in zip(chunks, plan):
        text = pl['text']
        said = pronounce(pl['say'], extra)
        speed = base_speed * pl['speed_k']
        key = hashlib.md5(json.dumps([KOKORO['name'], cfg.get('acento', 'es-419'), sid, said, round(speed, 4), pl['pitch'], VOICE_VERSION,
                                      checker is not None]).encode()).hexdigest()[:16]
        hit = cache / f'{key}.npy'
        if hit.exists():            # frase ya hecha antes (mismo texto y estilo): no se vuelve a sintetizar
            x = np.load(hit); meta = json.loads((cache / f'{key}.json').read_text())
            sc, missing, heard = meta['score'], meta['missing'], meta['heard']
        else:
            # ", " al inicio: el modelo arranca "en frío" y se comía el primer sonido ("¿Tu moto" → "Moto");
            # con la coma la frase arranca entera y el silencio se recorta después. Variantes más lentas si el
            # vigilante no entiende la frase.
            variants = [(', ' + said, speed), (', ' + said, speed * .93), ('... ' + said, speed * .88)]
            best = None
            for txt, sp in variants:
                a = tts.generate(txt, sid=sid, speed=sp)
                sr = a.sample_rate
                x = _trim(np.asarray(a.samples, np.float32), sr)
                if checker is None:
                    best = (1.0, [], '', x); break
                heard = checker.hear(x, sr)
                sc, missing = checker.score(text, heard)
                if best is None or sc > best[0]:
                    best = (sc, missing, heard, x)
                if sc >= .93:
                    break
            sc, missing, heard, x = best
            x = narrador.pitch_shift(x, sr, pl['pitch'])
            np.save(hit, x); (cache / f'{key}.json').write_text(json.dumps({'score': sc, 'missing': missing, 'heard': heard}))
        if checker is not None:
            report.append({'text': text, 'said': said, 'heard': heard, 'score': round(sc, 3), 'missing': missing,
                           'at': round(pos, 2), 'role': pl['role']})
        pb = pl['pause_before'] * float(cfg.get('pausas', 1)) if audio else 0
        if pb >= .15 and cfg.get('respiracion', True) is not False:      # respiro audible antes del suspenso
            b = narrador.breath(sr, min(.2, pb), seed=len(audio))
            audio += [np.zeros(int(max(0, pb - len(b) / sr) * sr), np.float32), b]
        elif pb:
            audio.append(np.zeros(int(pb * sr), np.float32))
        pos += pb
        dur = len(x) / sr
        words += estimate_words(ch, pos + .1, pos + dur - .12)
        pause = pl['pause_after'] * float(cfg.get('pausas', 1))
        audio += [x, np.zeros(int(pause * sr), np.float32)]
        pos += dur + pause
    pcm = (np.clip(np.concatenate(audio), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    if report:
        bad = [r for r in report if r['score'] < .85]
        log(f'  vigilante de voz: {len(report) - len(bad)}/{len(report)} oraciones se entienden bien')
        for r in bad:
            log(f'   ⚠ {r["at"]:5.1f} s "{r["text"]}" — se oye "{r["heard"]}" (faltan: {", ".join(r["missing"]) or "-"})')
    return words, pos, report


def whisper_words(path: Path, language: str = 'es') -> list[tuple[str, float, float]] | None:
    try:
        from faster_whisper import WhisperModel  # opcional
    except ImportError:
        return None
    model = WhisperModel('small', device='cpu', compute_type='int8')
    segs, _ = model.transcribe(str(path), language=language, word_timestamps=True)
    return [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in s.words]


def _edge_directed(toks: list[str], cfg: dict, out: Path, log=print):
    """Edge frase por frase con el narrador: cada frase con su ritmo, tono y silencios reales entre frases
    (Edge de corrido casi no hace pausas). Devuelve (palabras, duración)."""
    import tempfile
    import wave

    import numpy as np

    from . import narrador
    from .speech import pronounce
    chunks, cur = [], []
    for t in toks:
        cur.append(t)
        if t.endswith(('.', '?', '!', '…')):
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    chunks = narrador.beats(chunks)
    plan = narrador.plan([' '.join(c) for c in chunks])
    base = int(str(cfg.get('rate', '+0%')).rstrip('%') or 0)
    sr, audio, words, pos = 24000, [], [], 0.0
    with tempfile.TemporaryDirectory() as td:
        for i, (ch, pl) in enumerate(zip(chunks, plan)):
            mp3 = Path(td) / f'{i}.mp3'
            c2 = {**cfg, 'rate': f'{base + round((pl["speed_k"] - 1) * 100):+d}%', 'pitch': f'{round(pl["pitch"] * 9 * float(cfg.get("expresion", 1))):+d}Hz',
                  'volume': f'{pl.get("volume", 0):+d}%'}
            spoken = asyncio.run(_edge(pronounce(pl['say'], cfg.get('pronounce')) if cfg.get('pronounce') else pl['say'], c2, mp3))
            raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(mp3), '-f', 'f32le', '-ac', '1', '-ar', str(sr), '-'],
                                 capture_output=True, check=True).stdout
            # Edge llega limpia: márgenes cortos (los de Kokoro dejaban ~0,3 s extra de silencio por frase)
            x, a0 = _trim(np.frombuffer(raw, np.float32).copy(), sr, with_start=True, pad0=.08, pad1=.09)
            pk = float(cfg.get('pausas', 1))                # "pausas": .8 = silencios 20 % más cortos
            pb = pl['pause_before'] * pk if audio else 0
            if pb:
                audio.append(np.zeros(int(pb * sr), np.float32)); pos += pb
            dur = len(x) / sr
            sp = [(w, max(0, a - a0 / sr) + pos, max(0, b - a0 / sr) + pos) for w, a, b in spoken]
            words += align(ch, sp, pos + dur) if sp else estimate_words(ch, pos + .05, pos + dur - .1)
            audio += [x, np.zeros(int(pl['pause_after'] * pk * sr), np.float32)]
            pos += dur + pl['pause_after'] * pk
    pcm = (np.clip(np.concatenate(audio), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    return words, pos


async def _edge(text: str, cfg: dict, out: Path) -> list[tuple[str, float, float]]:
    import edge_tts
    voice = cfg['voice'] if 'Neural' in str(cfg.get('voice')) else 'es-CO-GonzaloNeural'   # si la marca usa Kokoro
    kw = dict(voice=voice, rate=cfg.get('rate', '+0%'), pitch=cfg.get('pitch', '+0Hz'), volume=cfg.get('volume', '+0%'))
    try:
        com = edge_tts.Communicate(text, boundary='WordBoundary', **kw)
    except TypeError:  # edge-tts < 7 siempre manda WordBoundary
        com = edge_tts.Communicate(text, **kw)
    spoken = []
    with open(out, 'wb') as f:
        async for ch in com.stream():
            if ch['type'] == 'audio':
                f.write(ch['data'])
            elif ch['type'] == 'WordBoundary':
                st = ch['offset'] / 1e7
                spoken.append((ch['text'], st, st + ch['duration'] / 1e7))
    return spoken


def synthesize(narration: str, cfg: dict, build: Path, base: Path | None = None, log=print) -> dict:
    """Devuelve {'audio': Path|None, 'duration': s, 'words': [Word], 'engine': str, 'timing': 'exact'|'whisper'|'estimated'}."""
    cfg = {**DEFAULT_VOICE, **(cfg or {})}
    engine = cfg['engine']
    vdir = build / 'voice'
    vdir.mkdir(parents=True, exist_ok=True)
    key_src = json.dumps({k: cfg.get(k) for k in ('engine', 'voice', 'rate', 'pitch', 'file', 'model', 'speed', 'pronounce', 'acento', 'narrador', 'timings', 'pausas', 'expresion')}, sort_keys=True) + narration + VOICE_VERSION
    key = hashlib.sha1(key_src.encode()).hexdigest()[:12]
    meta_path = vdir / f'{key}.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        audio = Path(meta['audio']) if meta.get('audio') else None
        if audio is not None and not audio.exists() and (vdir / audio.name).exists():
            audio = vdir / audio.name        # la carpeta se copió de otra máquina (render repartido)
        if audio is None or audio.exists():
            log(f'  voz: reutilizando {meta_path.name} ({meta["engine"]}, tiempos {meta["timing"]})')
            return {**meta, 'audio': audio, 'words': [Word(**w) for w in meta['words']]}
    toks = tokens(narration)
    timing = 'estimated'
    check = None
    audio: Path | None = None
    if engine == 'edge' and cfg.get('narrador', True) is not False:
        audio = vdir / f'{key}.wav'
        log(f'  voz: Edge TTS {cfg["voice"]} con narrador ...')
        try:
            words, dur = _edge_directed(toks, cfg, audio, log)
        except Exception as e:
            # sin internet a Microsoft (como en la sesión en la nube): se pide la misma voz al repositorio público
            # motocine-voz (GitHub la genera gratis) y se usa esa grabación con los tiempos de cada palabra
            if base is None or not (base / 'video.json').exists():
                raise RuntimeError(f'No se pudo generar la voz con Edge TTS ({e}). Revisa la conexión.') from e
            from .nube import run
            name = {'es-MX-JorgeNeural': 'edge-jorge-mx', 'es-CO-GonzaloNeural': 'edge-gonzalo-co'}.get(cfg.get('voice'), 'edge-jorge-mx')
            log(f'  voz: sin conexión a Edge aquí; la pido a GitHub (motocine-voz, {name})')
            mp3 = run(base, name, log=log)
            return synthesize(narration, {'engine': 'file', 'file': str(mp3), 'timings': str(mp3.with_suffix('.json')),
                                          'start': cfg.get('start')}, build, base, log)
        timing = 'exact'
    elif engine == 'edge':
        audio = vdir / f'{key}.mp3'
        log(f'  voz: Edge TTS {cfg["voice"]} ...')
        try:
            spoken = asyncio.run(_edge(narration, cfg, audio))
        except Exception as e:  # sin internet o servicio caído
            raise RuntimeError(f'No se pudo generar la voz con Edge TTS ({e}). Revisa la conexión, '
                               f'o usa --voice-engine silent para probar sin voz, o voice.engine "file" con tu grabación.') from e
        dur = ffprobe_duration(audio)
        if spoken:
            words, timing = align(toks, spoken, dur), 'exact'
        else:
            s, e = speech_bounds(audio)
            words = estimate_words(toks, s, e)
    elif engine in ('file', 'piper'):
        if engine == 'file':
            src = Path(cfg['file'])
            if base and not src.is_absolute():
                src = base / src
            if not src.exists():
                raise FileNotFoundError(f'No existe la grabación {src}')
            audio = vdir / f'{key}{src.suffix}'
            shutil.copy(src, audio)
        else:
            if not shutil.which('piper'):
                raise RuntimeError('No está instalado "piper". Instálalo con: pip install piper-tts')
            audio = vdir / f'{key}.wav'
            subprocess.run(['piper', '--model', str(cfg['model']), '--output_file', str(audio)], input=narration.encode(), check=True)
        dur = ffprobe_duration(audio)
        tm = cfg.get('timings')                       # tiempos ya medidos (p. ej. del laboratorio de voz: <voz>.json)
        if tm:
            tp = Path(tm) if not base or Path(tm).is_absolute() else base / tm
            spoken = [tuple(w) for w in json.loads(tp.read_text(encoding='utf-8'))['words']]
            words, timing = align(toks, spoken, dur), 'exact'
        elif spoken := whisper_words(audio):
            words, timing = align(toks, spoken, dur), 'whisper'
        else:
            s, e = speech_bounds(audio)
            words = estimate_words(toks, s, e)
    elif engine == 'kokoro':
        audio = vdir / f'{key}.wav'
        log(f'  voz: Kokoro {cfg.get("voice") if cfg.get("voice") in KOKORO["voices"] else "em_alex"} (local, sin internet) ...')
        words, dur, check = _kokoro(toks, cfg, audio, log)
        timing = 'phrase'
    elif engine == 'silent':
        speech = sum(len(t) for t in toks) * 0.062 + 0.35 * sum(t.endswith((',', '.', '?', '!', ':')) for t in toks)
        dur = speech + 0.3
        words = estimate_words(toks, 0.05, speech)
    else:
        raise ValueError(f'Motor de voz desconocido: {engine}')
    meta = {'audio': str(audio) if audio else None, 'duration': round(dur, 3), 'engine': engine, 'timing': timing,
            'voice': cfg.get('voice'), 'words': [w.to_dict() for w in words], 'check': check}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    log(f'  voz: {len(words)} palabras, {dur:.1f} s, tiempos {timing}')
    return {**meta, 'audio': audio, 'words': words}


def list_edge_voices(prefix: str = 'es-') -> list[dict]:
    import edge_tts
    voices = asyncio.run(edge_tts.list_voices())
    return [v for v in voices if v['ShortName'].startswith(prefix)]
