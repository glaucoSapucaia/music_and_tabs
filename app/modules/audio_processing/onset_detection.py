"""
Etapa 2 — Detecção de ataques (Onset Detection).

Identifica quando uma corda foi efetivamente tocada, distinguindo
o ataque do sustain/ressonância. Usa onset_detect do Librosa com
backtrack=True para recuar ao pico de energia real.

Saída: lista de timestamps (segundos) dos onsets detectados.
"""

from pathlib import Path

import numpy as np
import librosa
from loguru import logger


def detect_onsets(
    audio_path: str | Path,
    sr: int | None = None,
    hop_length: int = 512,
    backtrack: bool = True,
    delta: float = 0.07,
    wait: int = 10,
) -> dict:
    """
    Detecta onsets (ataques de corda) em um arquivo de áudio.

    Args:
        audio_path: Caminho para o áudio pré-processado
        sr: Sample rate (None = usa o original do arquivo)
        hop_length: Tamanho do hop em amostras. 512 @ 44100Hz ≈ 11.6ms por frame.
        backtrack: Recua cada onset ao pico de energia local anterior,
                   encontrando o ataque real em vez do pico de envelope.
        delta: Sensibilidade da detecção. Maior = menos onsets detectados
               (mais conservador). Range: 0.03–0.15.
        wait: Número de frames mínimos entre onsets consecutivos.
              wait=10 @ hop=512 @ 44100Hz ≈ 116ms mínimo entre notas.

    Returns:
        dict com 'onset_times' (lista de floats), 'onset_count',
        'mean_interval' e 'sample_rate'
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")

    logger.info(f"Detectando onsets: {audio_path.name}")
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    # Função de detecção de onset: combinação de flux espectral e energia RMS
    # onset_strength retorna um envelope de "novidade" ao longo do tempo
    onset_envelope = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,       # median é mais robusto que mean para violão
        fmax=1500.0,               # ignora harmônicos altos na detecção
    )

    # Detecta picos no envelope
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack,
        delta=delta,
        wait=wait,
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Filtra onsets muito próximos do fim do áudio (artefatos)
    max_time = len(y) / sr - 0.1
    onset_times = onset_times[onset_times < max_time]

    n = len(onset_times)
    if n > 1:
        intervals = np.diff(onset_times)
        mean_interval = float(np.mean(intervals))
        estimated_bpm = 60.0 / mean_interval if mean_interval > 0 else 0.0
    else:
        mean_interval = 0.0
        estimated_bpm = 0.0

    logger.info(
        f"Onsets detectados: {n} | "
        f"intervalo médio: {mean_interval*1000:.0f}ms | "
        f"BPM estimado: {estimated_bpm:.0f}"
    )

    return {
        "onset_times": onset_times.tolist(),
        "onset_count": n,
        "mean_interval": mean_interval,
        "estimated_bpm": estimated_bpm,
        "sample_rate": int(sr),
        "audio_duration": float(len(y) / sr),
    }
