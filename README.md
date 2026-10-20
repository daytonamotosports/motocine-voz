# MotoCine Voz

Laboratorio de voz en español latino para los videos de Daytona Moto Sport.

1. Sube un guion a `guiones/` (un archivo `.txt` con la narración).
2. GitHub genera la narración con varias voces (Kokoro local, Edge Jorge/Gonzalo con narrador automático,
   Chatterbox con emoción) y las califica:
   - **naturalidad** (UTMOS 1–5; una grabación humana normal saca ~4.0–4.4),
   - **claridad** (% de palabras que entiende Whisper),
   - **pausas** y **ritmo** (palabras por segundo).
3. Los audios y el informe quedan en `resultados/<guion>/` (`informe.md`).

El código de voz es una copia de `motocine/` del proyecto principal (narrador, pronunciación, vigilante).
