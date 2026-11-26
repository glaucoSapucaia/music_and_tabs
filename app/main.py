"""
music-and-tabs — Pipeline de produção audiovisual com transcrição musical.

Uso:
    uv run main.py transcribe input/audio.wav
    uv run main.py transcribe input/audio.wav --onset 0.4 --frame 0.3
"""

import json
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Setup de caminhos ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

# ── Carrega settings ─────────────────────────────────────────────────────────

def load_settings() -> dict:
    if not CONFIG_PATH.exists():
        typer.echo(f"[erro] settings.json não encontrado em: {CONFIG_PATH}", err=True)
        raise typer.Exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

# ── Configura loguru ─────────────────────────────────────────────────────────

def setup_logging(settings: dict) -> None:
    cfg = settings.get("logging", {})
    log_file = BASE_DIR / settings["paths"]["logs_dir"] / "app.log"
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("level", "INFO"))
    logger.add(
        str(log_file),
        level=cfg.get("level", "INFO"),
        rotation=cfg.get("rotation", "10 MB"),
        retention=cfg.get("retention", "30 days"),
        encoding="utf-8",
    )

# ── CLI ──────────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="music-tabs",
    help="Pipeline de transcrição musical e produção audiovisual.",
    add_completion=False,
)
console = Console()


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(
        ...,
        help="Caminho para o arquivo de áudio (.wav ou .mp3)",
        exists=True,
        readable=True,
    ),
    onset: float = typer.Option(None, "--onset", "-o", help="Onset threshold (0-1)"),
    frame: float = typer.Option(None, "--frame", "-f", help="Frame threshold (0-1)"),
    min_freq: float = typer.Option(None, "--min-freq", help="Frequência mínima em Hz"),
    max_freq: float = typer.Option(None, "--max-freq", help="Frequência máxima em Hz"),
) -> None:
    """Transcreve um arquivo de áudio e gera MIDI + JSON de notas."""
    settings = load_settings()
    setup_logging(settings)

    audio_cfg = settings["audio"]
    paths = settings["paths"]

    # Sobrescreve com valores da CLI se fornecidos
    onset_threshold = onset if onset is not None else audio_cfg["onset_threshold"]
    frame_threshold = frame if frame is not None else audio_cfg["frame_threshold"]
    minimum_frequency = min_freq if min_freq is not None else audio_cfg["minimum_frequency"]
    maximum_frequency = max_freq if max_freq is not None else audio_cfg["maximum_frequency"]

    console.print(Panel.fit(
        f"[bold cyan]Transcrevendo:[/bold cyan] {audio_file.name}\n"
        f"onset={onset_threshold}  frame={frame_threshold}  "
        f"freq=[{minimum_frequency}, {maximum_frequency}] Hz",
        title="music-and-tabs",
    ))

    from modules.audio_processing.transcription import transcribe as run_transcription

    result = run_transcription(
        audio_path=audio_file,
        output_midi_dir=BASE_DIR / paths["output_midi"],
        output_notes_dir=BASE_DIR / paths["output_notes"],
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=audio_cfg["minimum_note_length"],
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        melodia_trick=audio_cfg["melodia_trick"],
    )

    table = Table(title="Resultado da Transcrição", show_header=True)
    table.add_column("Campo", style="cyan")
    table.add_column("Valor", style="green")
    table.add_row("Notas detectadas", str(result["note_count"]))
    table.add_row("MIDI", result["midi_path"])
    table.add_row("JSON de notas", result["notes_path"])
    console.print(table)


if __name__ == "__main__":
    app()
