"""
Etapa 1 — Pré-processamento do áudio.

Normalização, redução de ruído espectral e filtragem passa-banda
para deixar o sinal ideal para detecção de notas de violão.

Faixa útil do violão: ~82 Hz (E2, 6ª corda solta) a ~1050 Hz (C6, 12º
traste da 1ª corda). O corte agressivo acima disso elimina harmônicos
que enganam o modelo de transcrição.
"""

from pathlib import Path

import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from loguru import logger


def preprocess(
    audio_path: str | Path,
    output_path: str | Path,
    lowcut: float = 75.0,
    highcut: float = 1500.0,
    order: int = 4,
    normalize: bool = True,
    reduce_noise: bool = True,
) -> dict:
    """
    Pré-processa o áudio para transcrição.

    Args:
        audio_path: Arquivo de entrada (.wav)
        output_path: Arquivo de saída (.wav processado)
        lowcut: Frequência de corte grave (Hz). Default 75 — abaixo de E2 (82Hz)
                já é ruído/handling do instrumento.
        highcut: Frequência de corte agudo (Hz). Default 1500 — cobre o range
                fundamental do violão com margem para o modelo.
        order: Ordem do filtro Butterworth.
        normalize: Se normaliza amplitude de pico.
        reduce_noise: Se aplica redução de ruído espectral.

    Returns:
        dict com 'output_path', 'duration', 'sample_rate' e 'snr_estimate'
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")

    logger.info(f"Carregando: {audio_path.name}")
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    logger.debug(f"Áudio: {len(y)/sr:.1f}s @ {sr}Hz")

    # ── 1. Redução de ruído ────────────────────────────────────────────────
    if reduce_noise:
        logger.info("Reduzindo ruído espectral...")
        y = nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=False,
            prop_decrease=0.85,   # remove até 85% do ruído estimado
        )

    # ── 2. Filtro passa-banda Butterworth (zero-phase) ──────────────────────
    logger.info(f"Aplicando passa-banda [{lowcut}, {highcut}] Hz...")
    sos = butter(order, [lowcut, highcut], btype="band", fs=sr, output="sos")
    y = sosfiltfilt(sos, y)

    # ── 3. Normalização de pico ────────────────────────────────────────────
    if normalize:
        peak = np.max(np.abs(y))
        if peak > 1e-8:
            y = y / peak * 0.95
        logger.debug(f"Normalizado: pico em {np.max(np.abs(y)):.3f}")

    # ── 4. Estimativa de SNR pós-processamento (dB) ─────────────────────────
    # Compara energia RMS do sinal com a dos trechos mais silenciosos
    rms_total = float(np.sqrt(np.mean(y ** 2)))
    frame_len = int(sr * 0.05)
    n_frames = len(y) // frame_len
    if n_frames > 0:
        rms_frames = np.array([
            np.sqrt(np.mean(y[i*frame_len:(i+1)*frame_len] ** 2))
            for i in range(n_frames)
        ])
        rms_quiet = float(np.percentile(rms_frames, 5))
        snr = 20 * np.log10(rms_total / (rms_quiet + 1e-10))
    else:
        snr = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), y, sr)
    logger.info(f"Áudio processado salvo em: {output_path} (SNR estimado: {snr:.1f} dB)")

    return {
        "output_path": str(output_path),
        "duration": float(len(y) / sr),
        "sample_rate": int(sr),
        "snr_estimate": float(snr),
    }
