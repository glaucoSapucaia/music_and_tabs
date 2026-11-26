"""
Etapa 3 — Identificação de notas por janela de onset.

Modo padrão: Basic Pitch analisa cada janela, aceita apenas notas
que começam na região de ataque (ATTACK_REGION segundos).

Modo onset-only (para tablatura): cada onset gera notas de duração
fixa. Onsets dentro de CHORD_CLUSTER_MS são agrupados como acorde
simultâneo — ideal para tab onde duração não importa.
"""

from pathlib import Path
import tempfile

import numpy as np
import librosa
import soundfile as sf
from loguru import logger

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict, Model


# Tamanho mínimo de janela em segundos (janelas menores são descartadas)
MIN_WINDOW_DURATION = 0.08

# Sem overlap entre janelas — a nota termina no onset seguinte
WINDOW_TAIL = 0.0

# Região de ataque: só notas que começam aqui são aceitas.
ATTACK_REGION = 0.08

# ── Modo onset-only ──────────────────────────────────────────────────────────
# Duração fixa atribuída a cada nota (ms) — só visual no MIDI/tab
FIXED_NOTE_DURATION = 0.25

# Janela de clustering: onsets dentro desse intervalo viram acorde simultâneo
CHORD_CLUSTER_MS = 0.060


def identify_notes_in_windows(
    audio_path: str | Path,
    onset_times: list[float],
    onset_threshold: float = 0.6,
    frame_threshold: float = 0.5,
    minimum_frequency: float = 82.0,
    maximum_frequency: float = 1050.0,
    max_notes_per_window: int = 6,
) -> list[dict]:
    """
    Identifica notas rodando o Basic Pitch em cada janela de onset.

    O modelo é carregado uma única vez antes do loop de janelas.

    Args:
        audio_path: Áudio pré-processado
        onset_times: Lista de timestamps de onsets (em segundos)
        onset_threshold: Sensibilidade de onset do Basic Pitch
        frame_threshold: Sensibilidade de frame do Basic Pitch
        minimum_frequency: Hz mínimo (E2 = 82 Hz para violão)
        maximum_frequency: Hz máximo
        max_notes_per_window: Máx notas simultâneas por janela (6 = nº de cordas)

    Returns:
        Lista de notas: { start_time, end_time, pitch_midi, amplitude }
        com timestamps absolutos em relação ao áudio original.
    """
    audio_path = Path(audio_path)
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    total_duration = len(y) / sr

    if not onset_times:
        logger.warning("Nenhum onset recebido — rodando em áudio completo como fallback")
        onset_times = [0.0]

    # ── Carrega o modelo uma única vez ──────────────────────────────────────
    logger.info("Carregando modelo Basic Pitch (uma vez)...")
    model = Model(ICASSP_2022_MODEL_PATH)
    logger.info("Modelo carregado.")

    # Monta janelas: [onset_i, onset_{i+1} + tail]
    windows = []
    for i, t_start in enumerate(onset_times):
        t_end = onset_times[i + 1] + WINDOW_TAIL if i + 1 < len(onset_times) else total_duration
        t_end = min(t_end, total_duration)
        if (t_end - t_start) >= MIN_WINDOW_DURATION:
            windows.append((t_start, t_end))

    n_windows = len(windows)
    logger.info(f"Processando {n_windows} janelas de onset...")

    all_notes: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (t_start, t_end) in enumerate(windows):
            s_start = int(t_start * sr)
            s_end   = int(t_end * sr)
            window_audio = y[s_start:s_end]

            if len(window_audio) < int(sr * MIN_WINDOW_DURATION):
                continue

            tmp_path = Path(tmpdir) / f"w_{idx:04d}.wav"
            sf.write(str(tmp_path), window_audio, sr)

            try:
                _, _, note_events = predict(
                    audio_path=str(tmp_path),
                    model_or_model_path=model,          # modelo já em memória
                    onset_threshold=onset_threshold,
                    frame_threshold=frame_threshold,
                    minimum_note_length=58,
                    minimum_frequency=minimum_frequency,
                    maximum_frequency=maximum_frequency,
                    melodia_trick=True,
                    multiple_pitch_bends=False,
                )
            except Exception as e:
                logger.warning(f"Janela {idx} ({t_start:.2f}s): {e}")
                continue

            # Corrige offset temporal para o áudio original
            # Só aceita notas que começam na região de ataque (primeiros ATTACK_REGION s)
            # Notas além disso são ressonância da nota anterior, não novos ataques.
            MIN_NOTE_DUR = 0.10
            window_notes = []
            for n in note_events:
                n_start = float(n[0]) + t_start
                n_end   = float(n[1]) + t_start

                # Descarta notas que começam fora da região de ataque
                if float(n[0]) > ATTACK_REGION:
                    continue

                if n_end <= n_start:
                    continue

                # Duração mínima garantida
                if (n_end - n_start) < MIN_NOTE_DUR:
                    n_end = n_start + MIN_NOTE_DUR

                # Não vaza além do fim da janela
                n_end = min(n_end, t_end) if t_end > n_start + MIN_NOTE_DUR else n_start + MIN_NOTE_DUR

                window_notes.append({
                    "start_time": round(n_start, 4),
                    "end_time":   round(n_end,   4),
                    "pitch_midi": int(n[2]),
                    "amplitude":  float(n[3]),
                })

            # Mantém apenas as mais fortes por janela (limite de polifonia)
            if len(window_notes) > max_notes_per_window:
                window_notes.sort(key=lambda n: n["amplitude"], reverse=True)
                window_notes = window_notes[:max_notes_per_window]

            all_notes.extend(window_notes)

            if (idx + 1) % 10 == 0 or (idx + 1) == n_windows:
                logger.debug(f"  {idx+1}/{n_windows} janelas | {len(all_notes)} notas até agora")

    all_notes.sort(key=lambda n: n["start_time"])
    logger.info(f"Total de notas identificadas: {len(all_notes)}")
    return all_notes


def identify_notes_onset_only(
    audio_path: str | Path,
    onset_times: list[float],
    onset_threshold: float = 0.6,
    frame_threshold: float = 0.5,
    minimum_frequency: float = 82.0,
    maximum_frequency: float = 1050.0,
    max_notes_per_attack: int = 6,
    chord_cluster_s: float = CHORD_CLUSTER_MS,
    fixed_duration_s: float = FIXED_NOTE_DURATION,
) -> list[dict]:
    """
    Modo tablatura: identifica apenas as notas tocadas em cada ataque,
    ignorando duração (todas recebem FIXED_NOTE_DURATION).

    Fluxo:
      1. Agrupa onsets em clusters (CHORD_CLUSTER_MS) — cada cluster
         é um ataque (possivelmente um acorde strummed)
      2. Para cada cluster, abre uma janela curta (do 1º onset do cluster
         até ATTACK_REGION após o último) e roda o Basic Pitch nela
      3. As notas detectadas na região de ataque ganham duração fixa
         e o timestamp do início do cluster

    Returns:
        Lista de notas: { start_time, end_time, pitch_midi, amplitude }
    """
    audio_path = Path(audio_path)
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    total_duration = len(y) / sr

    if not onset_times:
        logger.warning("Nenhum onset recebido")
        return []

    # ── 1. Clusteriza onsets ────────────────────────────────────────────────
    clusters: list[list[float]] = []
    current = [onset_times[0]]
    for t in onset_times[1:]:
        if t - current[-1] <= chord_cluster_s:
            current.append(t)
        else:
            clusters.append(current)
            current = [t]
    clusters.append(current)

    logger.info(f"{len(onset_times)} onsets → {len(clusters)} clusters de ataque")

    # ── 2. Carrega modelo uma vez ───────────────────────────────────────────
    model = Model(ICASSP_2022_MODEL_PATH)

    # ── 3. Processa cada cluster ────────────────────────────────────────────
    # Janela longa o suficiente para o Basic Pitch ter contexto (0.5s),
    # mas só notas que COMEÇAM na região de ataque são aceitas.
    CONTEXT_WINDOW = 0.5

    all_notes: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, cluster in enumerate(clusters):
            t_attack = cluster[0]              # tempo do ataque (início do cluster)
            t_end = min(t_attack + CONTEXT_WINDOW, total_duration)

            s_start = int(t_attack * sr)
            s_end   = int(t_end * sr)
            window_audio = y[s_start:s_end]

            if len(window_audio) < int(sr * MIN_WINDOW_DURATION):
                continue

            tmp_path = Path(tmpdir) / f"atk_{idx:04d}.wav"
            sf.write(str(tmp_path), window_audio, sr)

            try:
                _, _, note_events = predict(
                    audio_path=str(tmp_path),
                    model_or_model_path=model,
                    onset_threshold=onset_threshold,
                    frame_threshold=frame_threshold,
                    minimum_note_length=58,
                    minimum_frequency=minimum_frequency,
                    maximum_frequency=maximum_frequency,
                    melodia_trick=True,
                    multiple_pitch_bends=False,
                )
            except Exception as e:
                logger.warning(f"Cluster {idx} ({t_attack:.2f}s): {e}")
                continue

            # Notas aceitas: começam na região de ataque relativa à janela
            # A tolerância relativa é ATTACK_REGION + atraso entre cordas do cluster
            tolerance = ATTACK_REGION + (cluster[-1] - cluster[0])
            cluster_notes = []
            for n in note_events:
                if float(n[0]) > tolerance:
                    continue
                cluster_notes.append({
                    "start_time": round(t_attack, 4),
                    "end_time":   round(t_attack + fixed_duration_s, 4),
                    "pitch_midi": int(n[2]),
                    "amplitude":  float(n[3]),
                })

            # Limite de notas por ataque (uma por corda)
            if len(cluster_notes) > max_notes_per_attack:
                cluster_notes.sort(key=lambda n: n["amplitude"], reverse=True)
                cluster_notes = cluster_notes[:max_notes_per_attack]

            all_notes.extend(cluster_notes)

    all_notes.sort(key=lambda n: (n["start_time"], -n["pitch_midi"]))
    logger.info(f"Total de notas (modo onset-only): {len(all_notes)}")
    return all_notes
