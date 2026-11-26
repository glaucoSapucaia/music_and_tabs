"""
Etapa 5 — Pós-processamento musical.

Aplica correções musicais às notas identificadas:
  1. Remoção de duplicatas (mesma nota em < threshold ms)
  2. Limite de polifonia (máx 6 notas simultâneas — cordas do violão)
  3. Quantização rítmica opcional (snap para a grade mais próxima)
  4. Correção de notas fora do range do violão
  5. Remoção de notas muito curtas (< min_duration)
"""

import numpy as np
from loguru import logger


GUITAR_PITCH_MIN = 40  # E2 (corda solta com capo)
GUITAR_PITCH_MAX = 76  # E5 (limite prático do violão solo sem harmônicos)
MAX_POLYPHONY = 6      # número de cordas


def remove_duplicates(
    notes: list[dict],
    min_interval_ms: float = 50.0,
) -> list[dict]:
    """
    Remove notas duplicadas: mesma altura (pitch) com onset em menos de
    min_interval_ms ms de distância.
    """
    if not notes:
        return notes

    min_interval = min_interval_ms / 1000.0
    filtered = []
    # Agrupa por pitch e verifica proximidade temporal
    seen: dict[int, float] = {}  # pitch → último start_time

    for note in sorted(notes, key=lambda n: n["start_time"]):
        pitch = note["pitch_midi"]
        last = seen.get(pitch, -999.0)
        if note["start_time"] - last >= min_interval:
            filtered.append(note)
            seen[pitch] = note["start_time"]

    removed = len(notes) - len(filtered)
    if removed:
        logger.debug(f"Duplicatas removidas: {removed}")
    return filtered


def enforce_polyphony_limit(
    notes: list[dict],
    max_polyphony: int = MAX_POLYPHONY,
) -> list[dict]:
    """
    Garante no máximo max_polyphony notas simultâneas em qualquer instante.

    Percorre todos os eventos de início/fim em ordem cronológica, mantendo
    um conjunto de notas ativas. Quando o limite é excedido ao adicionar
    uma nota, remove a nota ativa mais fraca (menor amplitude).
    """
    if not notes:
        return notes

    # Cria eventos: (+1 = início, -1 = fim) com referência à nota
    events: list[tuple[float, int, int, dict]] = []
    for note in notes:
        events.append((note["start_time"],  0, id(note), note))   # 0 = start (processa primeiro)
        events.append((note["end_time"],    1, id(note), note))   # 1 = end

    events.sort(key=lambda e: (e[0], e[1]))

    active: list[dict] = []
    kept: set[int] = set()
    removed_ids: set[int] = set()

    for _, etype, nid, note in events:
        if etype == 1:
            # Fim de nota — remove dos ativos se estava
            active = [n for n in active if id(n) != nid]
        else:
            # Início de nota
            if id(note) in removed_ids:
                continue
            if len(active) < max_polyphony:
                active.append(note)
                kept.add(id(note))
            else:
                # Excedeu limite — compara com a mais fraca das ativas
                weakest = min(active, key=lambda n: n["amplitude"])
                if note["amplitude"] > weakest["amplitude"]:
                    active.remove(weakest)
                    removed_ids.add(id(weakest))
                    kept.discard(id(weakest))
                    active.append(note)
                    kept.add(id(note))
                else:
                    removed_ids.add(id(note))

    result = [n for n in notes if id(n) in kept]
    removed = len(notes) - len(result)
    if removed:
        logger.debug(f"Notas removidas por polifonia ({max_polyphony}): {removed}")
    return sorted(result, key=lambda n: n["start_time"])


def filter_range(
    notes: list[dict],
    pitch_min: int = GUITAR_PITCH_MIN,
    pitch_max: int = GUITAR_PITCH_MAX,
) -> list[dict]:
    """Remove notas fora do range real do violão."""
    filtered = [n for n in notes if pitch_min <= n["pitch_midi"] <= pitch_max]
    removed = len(notes) - len(filtered)
    if removed:
        logger.debug(f"Notas fora do range ({pitch_min}–{pitch_max}): {removed} removidas")
    return filtered


def filter_short_notes(
    notes: list[dict],
    min_duration_ms: float = 80.0,
) -> list[dict]:
    """Remove notas mais curtas que min_duration_ms."""
    min_dur = min_duration_ms / 1000.0
    filtered = [n for n in notes if (n["end_time"] - n["start_time"]) >= min_dur]
    removed = len(notes) - len(filtered)
    if removed:
        logger.debug(f"Notas < {min_duration_ms}ms removidas: {removed}")
    return filtered


def quantize(
    notes: list[dict],
    bpm: float,
    subdivision: int = 8,
    tolerance: float = 0.4,
) -> list[dict]:
    """
    Quantização rítmica: aproxima onsets à grade de subdivisões.

    Args:
        notes: Lista de notas
        bpm: BPM estimado (vindo do onset detector)
        subdivision: Subdivisão da grade (4=semínima, 8=colcheia, 16=semicolcheia)
        tolerance: Fração da subdivisão dentro da qual o snap é aplicado (0–1)
    """
    if bpm <= 0:
        return notes

    beat_duration = 60.0 / bpm
    grid = beat_duration / (subdivision / 4)
    threshold = grid * tolerance

    quantized = []
    for note in notes:
        t = note["start_time"]
        nearest = round(t / grid) * grid
        if abs(t - nearest) <= threshold:
            note = {**note, "start_time": round(nearest, 4)}
        quantized.append(note)

    return quantized


def merge_overlapping_same_pitch(notes: list[dict]) -> list[dict]:
    """
    Mescla notas do mesmo pitch que se sobrepõem ou estão a < 50ms de distância.
    A nota mesclada recebe a maior amplitude das duas.
    """
    if not notes:
        return notes

    from collections import defaultdict
    by_pitch: dict[int, list[dict]] = defaultdict(list)
    for n in notes:
        by_pitch[n["pitch_midi"]].append(n)

    result = []
    for pitch, group in by_pitch.items():
        group = sorted(group, key=lambda n: n["start_time"])
        merged = [group[0].copy()]
        for cur in group[1:]:
            prev = merged[-1]
            gap = cur["start_time"] - prev["end_time"]
            if gap < 0.05:  # sobreposição ou gap < 50ms → mescla
                prev["end_time"] = max(prev["end_time"], cur["end_time"])
                prev["amplitude"] = max(prev["amplitude"], cur["amplitude"])
            else:
                merged.append(cur.copy())
        result.extend(merged)

    removed = len(notes) - len(result)
    if removed:
        logger.debug(f"Notas sobrepostas mescladas: {removed}")
    return sorted(result, key=lambda n: n["start_time"])


def postprocess(
    notes: list[dict],
    bpm: float = 0.0,
    quantize_rhythm: bool = False,
    min_duration_ms: float = 80.0,
    min_duplicate_interval_ms: float = 50.0,
    max_polyphony: int = MAX_POLYPHONY,
) -> dict:
    """
    Aplica todas as etapas de pós-processamento em sequência.

    Returns:
        dict com 'notes' (lista processada) e contagens de cada etapa
    """
    original_count = len(notes)
    logger.info(f"Pós-processamento: {original_count} notas de entrada")

    notes = filter_range(notes)
    after_range = len(notes)

    notes = filter_short_notes(notes, min_duration_ms)
    after_short = len(notes)

    # Quantização ANTES da deduplicação — o snap pode criar duplicatas
    # (dois inícios próximos puxados para a mesma grade) que o dedup remove
    if quantize_rhythm and bpm > 0:
        notes = quantize(notes, bpm)
        logger.debug("Quantização rítmica aplicada")

    notes = remove_duplicates(notes, min_duplicate_interval_ms)
    after_dedup = len(notes)

    notes = merge_overlapping_same_pitch(notes)
    after_merge = len(notes)

    notes = enforce_polyphony_limit(notes, max_polyphony)
    after_poly = len(notes)

    logger.info(
        f"Pós-processamento concluído: {original_count} → {len(notes)} notas "
        f"(range: -{original_count - after_range}, "
        f"curtas: -{after_range - after_short}, "
        f"dedup: -{after_short - after_dedup}, "
        f"merge: -{after_dedup - after_merge}, "
        f"polifonia: -{after_merge - after_poly})"
    )

    return {
        "notes": notes,
        "stats": {
            "input": original_count,
            "after_range_filter": after_range,
            "after_short_filter": after_short,
            "after_deduplication": after_dedup,
            "after_polyphony_limit": after_poly,
            "output": len(notes),
        },
    }
