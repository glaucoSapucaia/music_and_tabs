"""
Módulo de detecção de acordes via chromagram (Librosa).

Estratégia:
  1. Extrai chromagram do áudio (energia por classe de nota: C, C#, D, ... B)
  2. Compara cada janela de tempo com templates de acordes conhecidos
  3. Agrupa janelas consecutivas com o mesmo acorde (smooth)
  4. Retorna sequência de acordes com timestamps

Saída:
  - JSON com lista de { chord, start_time, end_time, confidence }
  - MIDI com os acordes em blocos de notas simultâneas
"""

from pathlib import Path
import json

import numpy as np
import librosa
import pretty_midi
from loguru import logger


# ── Templates de acordes ─────────────────────────────────────────────────────
# Cada template é um vetor de 12 posições (C C# D D# E F F# G G# A A# B)
# com 1 nas notas que pertencem ao acorde e 0 nas demais.
# Baseado em tríades e tétrades mais comuns no violão.

def _build_templates() -> dict[str, np.ndarray]:
    """Constrói dicionário de templates de acordes normalizados."""

    # Notas: C=0 C#=1 D=2 D#=3 E=4 F=5 F#=6 G=7 G#=8 A=9 A#=10 B=11
    triades = {
        # Maiores
        "C":   [1,0,0,0,1,0,0,1,0,0,0,0],
        "C#":  [0,1,0,0,0,1,0,0,1,0,0,0],
        "D":   [0,0,1,0,0,0,1,0,0,1,0,0],
        "D#":  [0,0,0,1,0,0,0,1,0,0,1,0],
        "E":   [0,0,0,0,1,0,0,0,1,0,0,1],
        "F":   [1,0,0,0,0,1,0,0,0,1,0,0],
        "F#":  [0,1,0,0,0,0,1,0,0,0,1,0],
        "G":   [0,0,1,0,0,0,0,1,0,0,0,1],
        "G#":  [1,0,0,1,0,0,0,0,1,0,0,0],
        "A":   [0,1,0,0,1,0,0,0,0,1,0,0],
        "A#":  [0,0,1,0,0,1,0,0,0,0,1,0],
        "B":   [0,0,0,1,0,0,1,0,0,0,0,1],
        # Menores
        "Cm":  [1,0,0,1,0,0,0,1,0,0,0,0],
        "C#m": [0,1,0,0,1,0,0,0,1,0,0,0],
        "Dm":  [0,0,1,0,0,1,0,0,0,1,0,0],
        "D#m": [0,0,0,1,0,0,1,0,0,0,1,0],
        "Em":  [0,0,0,0,1,0,0,1,0,0,0,1],
        "Fm":  [1,0,0,0,0,1,0,0,1,0,0,0],
        "F#m": [0,1,0,0,0,0,1,0,0,1,0,0],
        "Gm":  [0,0,1,0,0,0,0,1,0,0,1,0],
        "G#m": [1,0,0,1,0,0,0,0,1,0,0,0],
        "Am":  [1,0,0,0,1,0,0,0,0,1,0,0],
        "A#m": [0,1,0,0,0,1,0,0,0,0,1,0],
        "Bm":  [0,0,1,0,0,0,1,0,0,0,0,1],
        # Dominantes com sétima (7)
        "C7":  [1,0,0,0,1,0,0,1,0,0,1,0],
        "D7":  [0,0,1,0,0,0,1,0,0,1,0,0],
        "E7":  [0,0,0,0,1,0,0,0,1,0,1,1],
        "G7":  [0,0,1,0,0,0,0,1,0,0,1,1],
        "A7":  [0,1,0,0,1,0,0,0,0,1,0,0],
        "B7":  [0,0,0,1,0,0,1,0,0,0,1,1],
        # Menores com sétima (m7)
        "Am7": [1,0,0,0,1,0,0,1,0,1,0,0],
        "Dm7": [0,0,1,0,0,1,0,0,0,1,0,0],
        "Em7": [0,0,0,0,1,0,0,1,0,0,1,1],
        # Suspensos comuns no violão
        "Dsus2": [0,0,1,0,0,0,1,0,0,1,0,0],
        "Dsus4": [0,0,1,0,0,1,0,1,0,0,0,0],
        "Asus2": [1,0,1,0,1,0,0,0,0,1,0,0],
        "Asus4": [0,0,0,0,1,0,0,0,0,1,0,1],
        "Esus4": [0,0,0,0,1,0,0,1,0,0,1,1],
        "Gsus2": [0,0,1,0,0,0,0,1,0,1,0,0],
    }

    templates = {}
    for name, pattern in triades.items():
        vec = np.array(pattern, dtype=float)
        norm = np.linalg.norm(vec)
        templates[name] = vec / norm if norm > 0 else vec

    return templates


CHORD_TEMPLATES = _build_templates()


# ── Funções principais ───────────────────────────────────────────────────────

def _chroma_to_chord(chroma_frame: np.ndarray) -> tuple[str, float]:
    """
    Compara um frame de chroma com todos os templates e retorna
    o acorde mais provável e sua confiança (similaridade de cosseno).
    """
    norm = np.linalg.norm(chroma_frame)
    if norm < 1e-6:
        return "N/C", 0.0  # Silêncio / sem acorde detectável

    chroma_norm = chroma_frame / norm
    best_chord = "N/C"
    best_score = -1.0

    for chord_name, template in CHORD_TEMPLATES.items():
        score = float(np.dot(chroma_norm, template))
        if score > best_score:
            best_score = score
            best_chord = chord_name

    return best_chord, best_score


def _smooth_sequence(
    chords: list[str],
    times: np.ndarray,
    min_duration: float = 0.5,
) -> list[dict]:
    """
    Agrupa frames consecutivos com o mesmo acorde e remove
    trocas muito rápidas (duração mínima configurável).

    Returns:
        Lista de { chord, start_time, end_time, confidence }
    """
    if not chords:
        return []

    segments = []
    current = chords[0]
    start_idx = 0

    for i in range(1, len(chords)):
        if chords[i] != current:
            segments.append({
                "chord": current,
                "start_time": float(times[start_idx]),
                "end_time": float(times[i]),
            })
            current = chords[i]
            start_idx = i

    # Último segmento
    segments.append({
        "chord": current,
        "start_time": float(times[start_idx]),
        "end_time": float(times[-1]),
    })

    # Remove segmentos abaixo da duração mínima (mescla com o anterior)
    filtered = []
    for seg in segments:
        dur = seg["end_time"] - seg["start_time"]
        if dur < min_duration and filtered:
            filtered[-1]["end_time"] = seg["end_time"]
        else:
            filtered.append(seg)

    return filtered


# ── Mapeamento acorde → notas MIDI ───────────────────────────────────────────
# Notas na oitava 3 (C3=48), posição de violão.
# Formato: lista de MIDI numbers que formam o acorde.

_C3 = 48  # dó central - 12

CHORD_NOTES: dict[str, list[int]] = {
    # Maiores (raiz + terça maior + quinta)
    "C":    [_C3,      _C3+4,  _C3+7],
    "C#":   [_C3+1,   _C3+5,  _C3+8],
    "D":    [_C3+2,   _C3+6,  _C3+9],
    "D#":   [_C3+3,   _C3+7,  _C3+10],
    "E":    [_C3+4,   _C3+8,  _C3+11],
    "F":    [_C3+5,   _C3+9,  _C3+12],
    "F#":   [_C3+6,   _C3+10, _C3+13],
    "G":    [_C3+7,   _C3+11, _C3+14],
    "G#":   [_C3+8,   _C3+12, _C3+15],
    "A":    [_C3+9,   _C3+13, _C3+16],
    "A#":   [_C3+10,  _C3+14, _C3+17],
    "B":    [_C3+11,  _C3+15, _C3+18],
    # Menores (raiz + terça menor + quinta)
    "Cm":   [_C3,      _C3+3,  _C3+7],
    "C#m":  [_C3+1,   _C3+4,  _C3+8],
    "Dm":   [_C3+2,   _C3+5,  _C3+9],
    "D#m":  [_C3+3,   _C3+6,  _C3+10],
    "Em":   [_C3+4,   _C3+7,  _C3+11],
    "Fm":   [_C3+5,   _C3+8,  _C3+12],
    "F#m":  [_C3+6,   _C3+9,  _C3+13],
    "Gm":   [_C3+7,   _C3+10, _C3+14],
    "G#m":  [_C3+8,   _C3+11, _C3+15],
    "Am":   [_C3+9,   _C3+12, _C3+16],
    "A#m":  [_C3+10,  _C3+13, _C3+17],
    "Bm":   [_C3+11,  _C3+14, _C3+18],
    # Dominantes com sétima
    "C7":   [_C3,      _C3+4,  _C3+7,  _C3+10],
    "D7":   [_C3+2,   _C3+6,  _C3+9,  _C3+12],
    "E7":   [_C3+4,   _C3+8,  _C3+11, _C3+14],
    "G7":   [_C3+7,   _C3+11, _C3+14, _C3+17],
    "A7":   [_C3+9,   _C3+13, _C3+16, _C3+19],
    "B7":   [_C3+11,  _C3+15, _C3+18, _C3+21],
    # Menores com sétima
    "Am7":  [_C3+9,   _C3+12, _C3+16, _C3+19],
    "Dm7":  [_C3+2,   _C3+5,  _C3+9,  _C3+12],
    "Em7":  [_C3+4,   _C3+7,  _C3+11, _C3+14],
    # Suspensos
    "Dsus2": [_C3+2,  _C3+4,  _C3+9],
    "Dsus4": [_C3+2,  _C3+7,  _C3+9],
    "Asus2": [_C3+9,  _C3+11, _C3+16],
    "Asus4": [_C3+9,  _C3+14, _C3+16],
    "Esus4": [_C3+4,  _C3+9,  _C3+11],
    "Gsus2": [_C3+7,  _C3+9,  _C3+14],
}


def export_chords_to_midi(
    segments: list[dict],
    output_path: str | Path,
    tempo: float = 120.0,
    velocity: int = 90,
) -> str:
    """
    Converte uma sequência de segmentos de acordes em arquivo MIDI.

    Cada acorde é inserido como um bloco de notas simultâneas com
    a duração exata do segmento detectado.

    Args:
        segments: Lista de { chord, start_time, end_time, confidence }
        output_path: Caminho do arquivo .mid a ser gerado
        tempo: BPM para o MIDI (não afeta o timing, apenas a grade de compassos)
        velocity: Velocidade (volume) das notas MIDI (0-127)

    Returns:
        Caminho do arquivo gerado
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Guitar (nylon)"),
        name="Chords",
    )

    for seg in segments:
        chord_name = seg["chord"]
        start = seg["start_time"]
        end = seg["end_time"]

        if chord_name == "N/C" or chord_name not in CHORD_NOTES:
            continue

        for pitch in CHORD_NOTES[chord_name]:
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=end,
            )
            instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(str(output_path))

    logger.info(f"MIDI de acordes salvo em: {output_path}")
    return str(output_path)


def detect_chords(
    audio_path: str | Path,
    output_dir: str | Path,
    output_stem: str | None = None,
    hop_length: int = 4096,
    min_segment_duration: float = 0.5,
    confidence_threshold: float = 0.0,
) -> dict:
    """
    Detecta acordes em um arquivo de áudio e salva o resultado em JSON.

    Args:
        audio_path: Caminho para o arquivo de áudio (.wav ou .mp3)
        output_dir: Diretório onde o JSON de acordes será salvo
        output_stem: Nome base do arquivo de saída (padrão: stem do áudio)
        hop_length: Tamanho do salto entre frames do chromagram.
                    Maior = janelas de tempo maiores, menos troca de acordes.
                    4096 ≈ ~93ms @ 44100Hz. Sugestão: 2048–8192.
        min_segment_duration: Duração mínima de um segmento de acorde em segundos.
                              Evita trocas muito rápidas causadas por transientes.
        confidence_threshold: Confiança mínima (0–1) para incluir um acorde.
                              Acordes abaixo disso aparecem como "N/C".

    Returns:
        dict com 'chords_path', 'chord_count' e 'unique_chords'
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem if output_stem else audio_path.stem
    chords_path = output_dir / f"{stem}_chords.json"

    logger.info(f"Carregando áudio: {audio_path.name}")
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    logger.debug(f"Áudio: {len(y)/sr:.1f}s @ {sr}Hz")

    logger.info("Extraindo chromagram...")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    # chroma shape: (12, n_frames) — cada coluna é um instante de tempo

    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
    )

    logger.info(f"Classificando acordes em {chroma.shape[1]} frames...")
    raw_chords = []
    raw_confidences = []

    for i in range(chroma.shape[1]):
        chord, confidence = _chroma_to_chord(chroma[:, i])
        if confidence < confidence_threshold:
            chord = "N/C"
        raw_chords.append(chord)
        raw_confidences.append(confidence)

    # Agrupa e suaviza a sequência
    segments = _smooth_sequence(raw_chords, frame_times, min_segment_duration)

    # Adiciona confiança média por segmento
    for seg in segments:
        idxs = [
            i for i, t in enumerate(frame_times)
            if seg["start_time"] <= t < seg["end_time"]
        ]
        if idxs:
            seg["confidence"] = round(float(np.mean([raw_confidences[i] for i in idxs])), 3)
        else:
            seg["confidence"] = 0.0

    unique_chords = sorted({s["chord"] for s in segments if s["chord"] != "N/C"})

    output_data = {
        "source_file": str(audio_path),
        "duration": float(len(y) / sr),
        "hop_length": hop_length,
        "min_segment_duration": min_segment_duration,
        "chord_count": len(segments),
        "unique_chords": unique_chords,
        "chords": segments,
    }

    with open(chords_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Acordes salvos em: {chords_path} "
        f"({len(segments)} segmentos, {len(unique_chords)} acordes únicos: {', '.join(unique_chords)})"
    )

    return {
        "chords_path": str(chords_path),
        "chord_count": len(segments),
        "unique_chords": unique_chords,
        "segments": segments,
    }


def detect_and_export(
    audio_path: str | Path,
    output_dir: str | Path,
    output_stem: str | None = None,
    hop_length: int = 4096,
    min_segment_duration: float = 0.5,
    confidence_threshold: float = 0.0,
    tempo: float = 120.0,
) -> dict:
    """
    Detecta acordes e exporta JSON + MIDI com a sequência detectada.

    Returns:
        dict com 'chords_path', 'midi_path', 'chord_count',
        'unique_chords' e 'segments'
    """
    result = detect_chords(
        audio_path=audio_path,
        output_dir=output_dir,
        output_stem=output_stem,
        hop_length=hop_length,
        min_segment_duration=min_segment_duration,
        confidence_threshold=confidence_threshold,
    )

    stem = output_stem if output_stem else Path(audio_path).stem
    midi_path = Path(output_dir) / f"{stem}_chords.mid"

    export_chords_to_midi(
        segments=result["segments"],
        output_path=midi_path,
        tempo=tempo,
    )

    result["midi_path"] = str(midi_path)
    return result
