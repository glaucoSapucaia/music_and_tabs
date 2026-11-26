"""
Métricas de qualidade da transcrição.

Calcula e persiste métricas que permitem comparar versões do pipeline
de forma objetiva, sem precisar ouvir o MIDI de cada rodada.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
from loguru import logger


GUITAR_PITCH_MIN = 40
GUITAR_PITCH_MAX = 76


def compute_metrics(
    notes: list[dict],
    onset_times: list[float],
    audio_duration: float,
    postprocess_stats: dict | None = None,
    snr_estimate: float | None = None,
    estimated_bpm: float | None = None,
) -> dict:
    """
    Calcula métricas de qualidade da transcrição.

    Args:
        notes: Lista de notas após pós-processamento
        onset_times: Onsets detectados na Etapa 2
        audio_duration: Duração total do áudio em segundos
        postprocess_stats: Contagens de cada etapa do pós-processamento
        snr_estimate: SNR do pré-processamento (dB)
        estimated_bpm: BPM estimado pelo onset detector

    Returns:
        dict com todas as métricas calculadas
    """
    if not notes:
        return {"error": "Nenhuma nota para calcular métricas"}

    pitches = [n["pitch_midi"] for n in notes]
    durations = [n["end_time"] - n["start_time"] for n in notes]
    amplitudes = [n["amplitude"] for n in notes]

    # ── Polifonia (simultaneidade instantânea, não pares de overlap) ────────
    # Cria eventos (+1 no início, -1 no fim) e percorre cronologicamente
    events = []
    for n in notes:
        events.append((n["start_time"], 1))
        events.append((n["end_time"], -1))
    events.sort(key=lambda e: (e[0], -e[1]))  # fim antes de início no mesmo t
    cur = 0
    max_poly = 0
    for _, delta in events:
        cur += delta
        max_poly = max(max_poly, cur)

    # Distribuição: polifonia no início de cada nota
    poly_counts: dict[int, int] = {}
    for note in notes:
        cur = sum(
            1 for m in notes
            if m["start_time"] <= note["start_time"] < m["end_time"]
        )
        poly_counts[cur] = poly_counts.get(cur, 0) + 1

    # ── Range e distribuição de pitch ───────────────────────────────────────
    out_of_range = sum(1 for p in pitches if p < GUITAR_PITCH_MIN or p > GUITAR_PITCH_MAX)

    # Notas por oitava (para ver se tem harmônicos espúrios)
    octave_dist: dict[str, int] = {}
    for p in pitches:
        octave = f"oct{p // 12 - 1}"
        octave_dist[octave] = octave_dist.get(octave, 0) + 1

    # ── Cobertura de onsets ──────────────────────────────────────────────────
    # Quantos onsets têm pelo menos uma nota dentro de 100ms
    covered = 0
    for t_onset in onset_times:
        has_note = any(
            abs(n["start_time"] - t_onset) < 0.1
            for n in notes
        )
        if has_note:
            covered += 1
    onset_coverage = covered / len(onset_times) if onset_times else 0.0

    # ── Consistência rítmica (desvio médio dos onsets para a grade) ─────────
    bpm = estimated_bpm or 0.0
    rhythm_consistency = None
    if bpm > 10:
        beat = 60.0 / bpm
        grid = beat / 2  # colcheias
        deviations = [
            abs(n["start_time"] % grid - round(n["start_time"] / grid) * grid % grid)
            for n in notes
        ]
        rhythm_consistency = float(np.mean(deviations) * 1000)  # em ms

    music_duration = max(n["end_time"] for n in notes) - min(n["start_time"] for n in notes)

    metrics = {
        # Contagem
        "note_count":            len(notes),
        "onset_count":           len(onset_times),
        "onset_coverage":        round(onset_coverage, 3),

        # Polifonia
        "max_polyphony":         max_poly,
        "polyphony_distribution": poly_counts,

        # Densidade
        "density_notes_per_sec": round(len(notes) / music_duration, 2) if music_duration > 0 else 0,

        # Range
        "pitch_min":             min(pitches),
        "pitch_max":             max(pitches),
        "pitch_out_of_range":    out_of_range,
        "octave_distribution":   octave_dist,

        # Duração das notas
        "duration_mean_ms":      round(float(np.mean(durations)) * 1000, 1),
        "duration_min_ms":       round(float(np.min(durations)) * 1000, 1),
        "duration_max_ms":       round(float(np.max(durations)) * 1000, 1),
        "notes_under_100ms":     sum(1 for d in durations if d < 0.099),

        # Amplitude
        "amplitude_mean":        round(float(np.mean(amplitudes)), 3),
        "amplitude_min":         round(float(np.min(amplitudes)), 3),

        # Ritmo
        "estimated_bpm":         round(bpm, 1) if bpm else None,
        "rhythm_consistency_ms": round(rhythm_consistency, 1) if rhythm_consistency else None,

        # Sinal
        "audio_duration_s":      round(audio_duration, 2),
        "snr_estimate_db":       round(snr_estimate, 1) if snr_estimate else None,

        # Pós-processamento
        "postprocess_stats":     postprocess_stats,
    }

    return metrics


def save_metrics(
    metrics: dict,
    output_path: str | Path,
    version_label: str = "",
    params: dict | None = None,
) -> str:
    """
    Persiste métricas em JSON com timestamp e parâmetros usados.

    Returns:
        Caminho do arquivo salvo
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "version": version_label,
        "timestamp": datetime.now().isoformat(),
        "params": params or {},
        "metrics": metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Métricas salvas em: {output_path}")
    return str(output_path)


def compare_metrics(paths: list[str | Path]) -> list[dict]:
    """
    Carrega e compara métricas de múltiplos arquivos.

    Returns:
        Lista de reports ordenados por version_label
    """
    reports = []
    for p in paths:
        p = Path(p)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                reports.append(json.load(f))

    return sorted(reports, key=lambda r: r.get("version", ""))
