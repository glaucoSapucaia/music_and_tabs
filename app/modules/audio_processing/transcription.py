"""
Módulo de transcrição de áudio para notas musicais usando Basic Pitch.

Recebe um arquivo de áudio (WAV/MP3) e gera:
- Arquivo MIDI com as notas detectadas
- JSON com dados brutos das notas (pitch, onset, offset, confidence)
"""

from pathlib import Path
import json

from loguru import logger
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH


def transcribe(
    audio_path: str | Path,
    output_midi_dir: str | Path,
    output_notes_dir: str | Path,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: int = 58,
    minimum_frequency: float = 65.0,
    maximum_frequency: float = 2100.0,
    melodia_trick: bool = True,
) -> dict:
    """
    Transcreve um arquivo de áudio e salva os resultados.

    Args:
        audio_path: Caminho para o arquivo de áudio (.wav ou .mp3)
        output_midi_dir: Diretório onde o arquivo MIDI será salvo
        output_notes_dir: Diretório onde o JSON de notas será salvo
        onset_threshold: Sensibilidade para detectar início das notas (0-1)
        frame_threshold: Sensibilidade para duração das notas (0-1)
        minimum_note_length: Duração mínima das notas em milissegundos
        minimum_frequency: Frequência mínima em Hz (E2 do violão ≈ 82 Hz)
        maximum_frequency: Frequência máxima em Hz
        melodia_trick: Melhora detecção de melodia principal

    Returns:
        dict com 'midi_path', 'notes_path' e 'note_count'
    """
    audio_path = Path(audio_path)
    output_midi_dir = Path(output_midi_dir)
    output_notes_dir = Path(output_notes_dir)

    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

    output_midi_dir.mkdir(parents=True, exist_ok=True)
    output_notes_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem
    midi_path = output_midi_dir / f"{stem}.mid"
    notes_path = output_notes_dir / f"{stem}_notes.json"

    logger.info(f"Iniciando transcrição: {audio_path.name}")
    logger.debug(
        f"Parâmetros — onset: {onset_threshold}, frame: {frame_threshold}, "
        f"freq: [{minimum_frequency}, {maximum_frequency}] Hz"
    )

    model_output, midi_data, note_events = predict(
        audio_path=str(audio_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        melodia_trick=melodia_trick,
        multiple_pitch_bends=False,
    )

    # Salva MIDI
    midi_data.write(str(midi_path))
    logger.info(f"MIDI salvo em: {midi_path}")

    # Serializa note_events para JSON
    notes_list = []
    for note in note_events:
        notes_list.append({
            "start_time": float(note[0]),
            "end_time": float(note[1]),
            "pitch_midi": int(note[2]),
            "amplitude": float(note[3]),
            "pitch_bends": [float(b) for b in note[4]] if note[4] is not None else [],
        })

    notes_data = {
        "source_file": str(audio_path),
        "note_count": len(notes_list),
        "notes": notes_list,
    }

    with open(notes_path, "w", encoding="utf-8") as f:
        json.dump(notes_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Notas salvas em: {notes_path} ({len(notes_list)} notas detectadas)")

    return {
        "midi_path": str(midi_path),
        "notes_path": str(notes_path),
        "note_count": len(notes_list),
    }
