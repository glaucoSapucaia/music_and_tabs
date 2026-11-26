"""
Pipeline completo de processamento de áudio: WAV → MIDI de alta fidelidade.

Etapas:
  1. Pré-processamento (preprocessor.py)
  2. Detecção de onsets (onset_detection.py)
  3. Identificação de notas por janela (note_identification.py)
  4. Segmentação (incorporada na etapa 3 via janelas)
  5. Pós-processamento (postprocessor.py)
  6. Métricas (metrics.py)

Saídas em output/processed/<stem>/:
  - <stem>_processed.wav   (áudio tratado)
  - <stem>_notes.json       (notas pós-processadas)
  - <stem>.mid              (MIDI final)
  - <stem>_metrics.json     (relatório de métricas)
"""

from pathlib import Path
import json

import pretty_midi
import librosa
from loguru import logger

from modules.audio_processing.preprocessor import preprocess
from modules.audio_processing.onset_detection import detect_onsets
from modules.audio_processing.note_identification import identify_notes_in_windows
from modules.audio_processing.postprocessor import postprocess
from modules.audio_processing.metrics import compute_metrics, save_metrics


def run_pipeline(
    audio_path: str | Path,
    output_dir: str | Path,
    output_stem: str | None = None,
    version_label: str = "v1",
    # Parâmetros do pré-processamento
    skip_preprocess: bool = False,
    lowcut: float = 75.0,
    highcut: float = 1500.0,
    reduce_noise: bool = True,
    # Parâmetros do onset detection
    onset_delta: float = 0.09,
    onset_wait: int = 10,
    # Parâmetros do Basic Pitch por janela
    bp_onset_threshold: float = 0.6,
    bp_frame_threshold: float = 0.5,
    min_frequency: float = 82.0,
    max_frequency: float = 1050.0,
    # Modo tablatura: ignora duração, agrupa onsets próximos como acorde
    onset_only: bool = False,
    chord_cluster_s: float = 0.060,
    fixed_note_duration_s: float = 0.25,
    # Parâmetros do pós-processamento
    quantize_rhythm: bool = True,
    min_note_duration_ms: float = 150.0,
    max_polyphony: int = 6,
    # MIDI
    tempo: float = 120.0,
) -> dict:
    """
    Executa o pipeline completo de processamento.

    Returns:
        dict com caminhos das saídas, métricas e estatísticas
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = output_stem if output_stem else audio_path.stem
    processed_wav = output_dir / f"{stem}_processed.wav"
    notes_json = output_dir / f"{stem}_notes.json"
    midi_path = output_dir / f"{stem}.mid"
    metrics_path = output_dir / f"{stem}_metrics.json"

    params = {
        "lowcut": lowcut, "highcut": highcut, "reduce_noise": reduce_noise,
        "onset_delta": onset_delta, "onset_wait": onset_wait,
        "bp_onset_threshold": bp_onset_threshold,
        "bp_frame_threshold": bp_frame_threshold,
        "min_frequency": min_frequency, "max_frequency": max_frequency,
        "quantize_rhythm": quantize_rhythm,
        "min_note_duration_ms": min_note_duration_ms,
        "max_polyphony": max_polyphony,
        "tempo": tempo,
    }

    # ══ Etapa 1: Pré-processamento ══════════════════════════════════════════
    logger.info(f"── Etapa 1/5: Pré-processamento ({version_label}) ──")

    if skip_preprocess:
        import shutil
        shutil.copy2(str(audio_path), str(processed_wav))
        pre_result = {
            "output_path": str(processed_wav),
            "duration": float(librosa.get_duration(path=str(audio_path))),
            "sample_rate": librosa.get_samplerate(str(audio_path)),
            "snr_estimate": None,
        }
        logger.info("Pré-processamento ignorado — usando áudio bruto")
    else:
        pre_result = preprocess(
            audio_path=audio_path,
            output_path=processed_wav,
            lowcut=lowcut,
            highcut=highcut,
            reduce_noise=reduce_noise,
        )

    # ══ Etapa 2: Detecção de onsets ═════════════════════════════════════════
    logger.info("── Etapa 2/5: Detecção de onsets ──")
    onset_result = detect_onsets(
        audio_path=processed_wav,
        delta=onset_delta,
        wait=onset_wait,
    )
    # ══ Etapa 3: Identificação de notas por janela ═══════════════════════════
    logger.info("── Etapa 3/5: Identificação de notas ──")

    if onset_only:
        logger.info("MODO TABLATURA: onset-only com clustering de acordes")
        from modules.audio_processing.note_identification import identify_notes_onset_only
        raw_notes = identify_notes_onset_only(
            audio_path=processed_wav,
            onset_times=onset_result["onset_times"],
            onset_threshold=bp_onset_threshold,
            frame_threshold=bp_frame_threshold,
            minimum_frequency=min_frequency,
            maximum_frequency=max_frequency,
            max_notes_per_attack=max_polyphony,
            chord_cluster_s=chord_cluster_s,
            fixed_duration_s=fixed_note_duration_s,
        )
    else:
        raw_notes = identify_notes_in_windows(
            audio_path=processed_wav,
            onset_times=onset_result["onset_times"],
            onset_threshold=bp_onset_threshold,
            frame_threshold=bp_frame_threshold,
            minimum_frequency=min_frequency,
            maximum_frequency=max_frequency,
            max_notes_per_window=max_polyphony,
        )

    # ══ Etapa 4/5: Pós-processamento ════════════════════════════════════════
    logger.info("── Etapa 4/5: Pós-processamento musical ──")

    if onset_only:
        from modules.audio_processing.postprocessor import (
            filter_range, enforce_polyphony_limit,
        )
        raw_notes = filter_range(raw_notes)
        raw_notes = enforce_polyphony_limit(raw_notes, max_polyphony)
        notes = raw_notes
        post_stats = {"input": len(raw_notes), "output": len(notes)}
    else:
        post_result = postprocess(
            notes=raw_notes,
            bpm=onset_result["estimated_bpm"],
            quantize_rhythm=quantize_rhythm,
            min_duration_ms=min_note_duration_ms,
            max_polyphony=max_polyphony,
        )
        notes = post_result["notes"]
        post_stats = post_result["stats"]

    # ══ Etapa 5: Exportação MIDI ════════════════════════════════════════════
    logger.info("── Etapa 5/5: Exportação MIDI e métricas ──")
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Guitar (nylon)"),
        name=f"guitar_{version_label}",
    )

    for note in notes:
        if note["end_time"] <= note["start_time"]:
            continue
        instrument.notes.append(pretty_midi.Note(
            velocity=max(40, min(127, int(note["amplitude"] * 127))),
            pitch=note["pitch_midi"],
            start=note["start_time"],
            end=note["end_time"],
        ))

    midi.instruments.append(instrument)
    midi.write(str(midi_path))
    logger.info(f"MIDI salvo: {midi_path}")

    # Salva notas processadas em JSON
    with open(notes_json, "w", encoding="utf-8") as f:
        json.dump({
            "source_file": str(audio_path),
            "version": version_label,
            "params": params,
            "note_count": len(notes),
            "notes": notes,
        }, f, indent=2, ensure_ascii=False)

    # ══ Métricas ════════════════════════════════════════════════════════════
    metrics = compute_metrics(
        notes=notes,
        onset_times=onset_result["onset_times"],
        audio_duration=pre_result["duration"],
        postprocess_stats=post_stats,
        snr_estimate=pre_result["snr_estimate"],
        estimated_bpm=onset_result["estimated_bpm"],
    )
    save_metrics(
        metrics=metrics,
        output_path=metrics_path,
        version_label=version_label,
        params=params,
    )

    return {
        "processed_wav": str(processed_wav),
        "notes_json": str(notes_json),
        "midi_path": str(midi_path),
        "metrics_path": str(metrics_path),
        "note_count": len(notes),
        "onset_count": onset_result["onset_count"],
        "estimated_bpm": onset_result["estimated_bpm"],
        "metrics": metrics,
        "params": params,
    }
