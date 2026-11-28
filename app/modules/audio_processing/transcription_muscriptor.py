"""
Módulo de transcrição de áudio para MIDI usando MuScriptor (Kyutai/Mirelo).

MuScriptor é um transformer decoder de 103M–1.4B parâmetros, significativamente
mais preciso que o Basic Pitch para transcrição de instrumentos reais.

Requer autenticação no HuggingFace (os pesos são CC BY-NC 4.0 — uso não-comercial):
  $ uvx hf auth login
  ou: export HF_TOKEN=hf_...

Modelos disponíveis:
  - small  (~210 MB) — prático em CPU
  - medium (~620 MB) — padrão, melhor trade-off velocidade/qualidade
  - large  (~2.7 GB) — máxima qualidade, recomenda GPU

Referência: https://github.com/muscriptor/muscriptor
"""

from pathlib import Path

from loguru import logger


def transcribe(
    audio_path: str | Path,
    output_midi_dir: str | Path,
    output_stem: str | None = None,
    model_size: str = "small",
    device: str | None = None,
    instruments: list[str] | None = None,
    detect_tempo: bool | str = "best-effort",
) -> dict:
    """
    Transcreve um arquivo de áudio para MIDI usando MuScriptor.

    Args:
        audio_path: Caminho para o arquivo de áudio (.wav, .mp3, .flac, etc.)
        output_midi_dir: Diretório onde o arquivo MIDI será salvo
        output_stem: Nome base do arquivo de saída (padrão: stem do áudio)
        model_size: Tamanho do modelo — 'small', 'medium' ou 'large'
        device: Dispositivo torch ('cpu', 'cuda', 'mps'). None = automático
        instruments: Lista de instrumentos para transcrever, ex: ['guitar', 'piano'].
                     None = detecção automática pelo modelo
        detect_tempo: Detecta BPM e escreve mapa de tempo no MIDI.
                      True | False | 'best-effort' (padrão)

    Returns:
        dict com 'midi_path' e 'model_size'

    Raises:
        FileNotFoundError: se o arquivo de áudio não existir
        RuntimeError: se o HuggingFace não estiver autenticado
    """
    from muscriptor import TranscriptionModel

    audio_path = Path(audio_path)
    output_midi_dir = Path(output_midi_dir)

    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

    output_midi_dir.mkdir(parents=True, exist_ok=True)

    stem = output_stem if output_stem else audio_path.stem
    midi_path = output_midi_dir / f"{stem}.mid"

    logger.info(f"Carregando modelo MuScriptor '{model_size}'...")
    logger.debug(f"device={device or 'auto'}  instruments={instruments or 'auto'}")

    try:
        model = TranscriptionModel.load_model(
            weights_path=model_size,
            device=device,
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "token" in msg.lower() or "login" in msg.lower() or "gated" in msg.lower():
            raise RuntimeError(
                "Autenticação no HuggingFace necessária.\n"
                "Execute: uvx hf auth login\n"
                "Ou defina: export HF_TOKEN=hf_...\n"
                f"Erro original: {msg}"
            ) from e
        raise

    logger.info(f"Transcrevendo: {audio_path.name}")

    midi_bytes = model.transcribe_to_midi(
        audio=str(audio_path),
        instruments=instruments,
        detect_tempo=detect_tempo,
    )

    with open(midi_path, "wb") as f:
        f.write(midi_bytes)

    logger.info(f"MIDI salvo em: {midi_path}")

    return {
        "midi_path": str(midi_path),
        "model_size": model_size,
    }
