"""Laboratorio de voz: genera la MISMA narración con varias voces y las califica con números, para elegir sin
depender del oído de nadie (y sin gastar: corre gratis en GitHub Actions, que sí tiene internet).

  python scripts/voz_lab.py projects/gasolina-iso lab-voz/

Calificaciones por voz (informe.md / informe.json):
  naturalidad  UTMOS (1–5): modelo entrenado con miles de calificaciones de personas. Una grabación humana
               normal saca ~4.0–4.4; voces robóticas < 3.5.
  entiende     % de palabras que un reconocedor (Whisper) entiende del guion.
  pausas       silencios de más de 0,25 s y su promedio (¿respira, deja suspenso?).
  ritmo        palabras por segundo (lo cómodo en redes: 2,5–3,2).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motocine.voice import synthesize  # noqa: E402

SR = 24000


def load(path: Path, sr: int = SR) -> np.ndarray:
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le', '-ac', '1', '-ar', str(sr), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def save_mp3(x: np.ndarray, out: Path, sr: int = SR):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(sr), '-ac', '1', '-i', '-',
                    '-af', 'loudnorm=I=-16:TP=-1.5', '-b:a', '128k', str(out)], input=x.astype(np.float32).tobytes(), check=True)


def pauses(x: np.ndarray, sr: int = SR) -> dict:
    win = int(.02 * sr)
    env = np.sqrt(np.convolve(x * x, np.ones(win) / win, 'same'))[::win]
    quiet = env < max(.004, env.max() * .02)
    runs, n = [], 0
    for q in quiet:
        if q:
            n += 1
        else:
            if n * .02 >= .25:
                runs.append(n * .02)
            n = 0
    return {'pausas': len(runs), 'pausa_media': round(float(np.mean(runs)), 2) if runs else 0.0}


_UTMOS = None


def utmos(x: np.ndarray, sr: int = SR) -> float | None:
    global _UTMOS
    try:
        import torch
        if _UTMOS is None:
            _UTMOS = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)
        y = load_resampled(x, sr, 16000)
        scores = []
        for i in range(0, len(y), 16000 * 10):              # de a 10 s (el modelo fue hecho para frases)
            seg = y[i:i + 16000 * 10]
            if len(seg) > 16000 * 2:
                scores.append(float(_UTMOS(torch.from_numpy(seg).unsqueeze(0), 16000)))
        return round(float(np.mean(scores)), 2) if scores else None
    except Exception:
        traceback.print_exc()
        return None


def load_resampled(x, sr, to):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(sr), '-ac', '1', '-i', '-', '-f', 'f32le',
                          '-ar', str(to), '-'], input=x.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


_CK = None


def understood(x: np.ndarray, text: str, sr: int = SR) -> tuple[float, str]:
    global _CK
    from motocine.speech import Checker
    if _CK is None:
        _CK = Checker(download=True)
    heard = []
    y = x
    for i in range(0, len(y), sr * 25):                   # Whisper escucha de a 30 s como máximo
        heard.append(_CK.hear(y[i:i + sr * 25], sr))
    h = ' '.join(heard)
    sc, _ = _CK.score(text, h)
    return round(sc, 3), h


def chatterbox(narration: str, ref: Path, out: Path, exaggeration: float = .6):
    """Chatterbox multilingüe (código abierto, licencia MIT): copia el timbre de una grabación corta de referencia y
    actúa con emoción. Se dice frase por frase con las pausas del narrador."""
    import torch  # noqa: F401
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    from motocine import narrador
    from motocine.text import tokens
    model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
    toks, chunks, cur = tokens(narration), [], []
    for t in toks:
        cur.append(t)
        if t.endswith(('.', '?', '!', '…')):
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    chunks = narrador.beats(chunks)
    plan = narrador.plan([' '.join(c) for c in chunks])
    parts, words, pos = [], [], 0.0
    from motocine.text import estimate_words
    for ch, pl in zip(chunks, plan):
        if pl['pause_before'] and parts:
            parts.append(np.zeros(int(pl['pause_before'] * SR), np.float32)); pos += pl['pause_before']
        w = model.generate(pl['say'], language_id='es', audio_prompt_path=str(ref), exaggeration=exaggeration,
                           cfg_weight=.5 if pl['role'] != 'intriga' else .35)
        y = w.squeeze().cpu().numpy().astype(np.float32)
        y = load_resampled(y, model.sr, SR)
        from motocine.voice import _trim
        y = _trim(y, SR)
        words += [[w.text, round(w.start, 3), round(w.end, 3)] for w in estimate_words(ch, pos + .08, pos + len(y) / SR - .12)]
        parts += [y, np.zeros(int(pl['pause_after'] * SR), np.float32)]
        pos += len(y) / SR + pl['pause_after']
    x = np.concatenate(parts)
    save_mp3(x, out)
    return x, words


def main(project: str, out_dir: str):
    proj = Path(project)                          # carpeta de proyecto (video.json) o un guion .txt
    narration = proj.read_text().strip() if proj.suffix == '.txt' else json.loads((proj / 'video.json').read_text())['narration']
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    build = Path('/tmp/vozlab'); build.mkdir(exist_ok=True)
    variants = {
        'kokoro-alex': {'engine': 'kokoro', 'voice': 'em_alex', 'speed': 1.15, 'check': 'download'},
        'edge-jorge-mx': {'engine': 'edge', 'voice': 'es-MX-JorgeNeural'},
        'edge-gonzalo-co': {'engine': 'edge', 'voice': 'es-CO-GonzaloNeural'},
        'edge-jorge-mx-sin-narrador': {'engine': 'edge', 'voice': 'es-MX-JorgeNeural', 'narrador': False},
    }
    rep, audio, timings = {}, {}, {}
    for name, cfg in variants.items():
        t0 = time.time()
        try:
            v = synthesize(narration, cfg, build / name)
            audio[name] = load(Path(v['audio']))
            timings[name] = [[w.text, round(w.start, 3), round(w.end, 3)] for w in v['words']]
            rep[name] = {'segundos_generar': round(time.time() - t0, 1)}
        except Exception as e:
            rep[name] = {'error': str(e)[:300]}
    if 'edge-jorge-mx' in audio:                       # referencia de timbre para Chatterbox (8 s de Jorge)
        ref = build / 'ref.wav'
        x = audio['edge-jorge-mx'][:SR * 9]
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-', str(ref)],
                       input=x.tobytes(), check=True)
        for ex in (.5, .8):
            name = f'chatterbox-emocion-{ex}'
            t0 = time.time()
            try:
                audio[name], timings[name] = chatterbox(narration, ref, out / f'{name}.mp3', ex)
                rep[name] = {'segundos_generar': round(time.time() - t0, 1)}
            except Exception as e:
                traceback.print_exc()
                rep[name] = {'error': str(e)[:300]}
    for name, x in audio.items():
        save_mp3(x, out / f'{name}.mp3')
        # tiempos de cada palabra (para usar esta voz en un video: subtítulos y boca del mecánico)
        (out / f'{name}.json').write_text(json.dumps({'narration': narration, 'words': timings.get(name, [])}, ensure_ascii=False))
        sc, heard = understood(x, narration)
        words = len(narration.split())
        rep[name].update({'naturalidad_utmos': utmos(x), 'entiende': sc, 'duracion': round(len(x) / SR, 1),
                          'palabras_por_seg': round(words / (len(x) / SR), 2), **pauses(x), 'se_oye': heard})
    (out / 'informe.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    lines = ['| voz | naturalidad (1-5) | entiende | pausas | pausa media | palabras/s | dura |', '|---|---|---|---|---|---|---|']
    for n, r in sorted(rep.items(), key=lambda kv: -(kv[1].get('naturalidad_utmos') or 0)):
        if 'error' in r:
            lines.append(f'| {n} | error: {r["error"][:80]} |||||')
        else:
            lines.append(f'| {n} | {r["naturalidad_utmos"]} | {r["entiende"]} | {r["pausas"]} | {r["pausa_media"]} | '
                         f'{r["palabras_por_seg"]} | {r["duracion"]} |')
    (out / 'informe.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(*sys.argv[1:3])
