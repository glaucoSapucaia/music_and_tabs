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
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Setup de caminhos ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

# Carrega .env da raiz do app (antes de qualquer outra coisa)
load_dotenv(BASE_DIR / ".env")
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
    engine: str = typer.Option(
        "muscriptor", "--engine", "-e",
        help="Engine de transcrição: 'muscriptor' (padrão) ou 'basic-pitch'",
    ),
    output_name: str = typer.Option(
        None, "--output-name", "-n",
        help="Nome base dos arquivos de saída (sem extensão). Padrão: nome do áudio.",
    ),
    # ── Opções MuScriptor ────────────────────────────────────────────────────
    model_size: str = typer.Option(
        None, "--model",
        help="[muscriptor] Tamanho do modelo: 'small', 'medium' ou 'large'",
    ),
    device: str = typer.Option(
        None, "--device",
        help="[muscriptor] Dispositivo: 'cpu', 'cuda', 'mps'. Padrão: automático",
    ),
    instruments: str = typer.Option(
        None, "--instruments",
        help="[muscriptor] Instrumentos separados por vírgula, ex: 'guitar,piano'. Padrão: automático",
    ),
    # ── Opções Basic Pitch ───────────────────────────────────────────────────
    onset: float = typer.Option(None, "--onset", "-o", help="[basic-pitch] Onset threshold (0-1)"),
    frame: float = typer.Option(None, "--frame", "-f", help="[basic-pitch] Frame threshold (0-1)"),
    min_freq: float = typer.Option(None, "--min-freq", help="[basic-pitch] Frequência mínima em Hz"),
    max_freq: float = typer.Option(None, "--max-freq", help="[basic-pitch] Frequência máxima em Hz"),
    min_note_len: int = typer.Option(None, "--min-note-len", help="[basic-pitch] Duração mínima de nota em ms"),
    no_melodia: bool = typer.Option(False, "--no-melodia", help="[basic-pitch] Desativa o melodia trick"),
) -> None:
    """Transcreve um arquivo de áudio e gera MIDI.

    Engine padrão: muscriptor (requer autenticação no HuggingFace).
    Use --engine basic-pitch para o engine alternativo sem autenticação.
    """
    settings = load_settings()
    setup_logging(settings)

    stem = output_name if output_name else audio_file.stem

    if engine not in ("muscriptor", "basic-pitch"):
        console.print(f"[red]Engine inválido: '{engine}'. Use 'muscriptor' ou 'basic-pitch'.[/red]")
        raise typer.Exit(1)

    # ── MuScriptor ───────────────────────────────────────────────────────────
    if engine == "muscriptor":
        musc_cfg = settings.get("muscriptor", {})
        size = model_size or musc_cfg.get("model_size", "small")
        dev  = device     or musc_cfg.get("device", None)
        instr_list = [i.strip() for i in instruments.split(",")] if instruments else None

        console.print(Panel.fit(
            f"[bold cyan]Transcrevendo:[/bold cyan] {audio_file.name}  →  [green]{stem}[/green]\n"
            f"engine=[bold]muscriptor[/bold]  model={size}  device={dev or 'auto'}"
            + (f"  instruments={','.join(instr_list)}" if instr_list else ""),
            title="music-and-tabs",
        ))

        from modules.audio_processing.transcription_muscriptor import transcribe as run_muscriptor

        try:
            result = run_muscriptor(
                audio_path=audio_file,
                output_midi_dir=BASE_DIR / settings["paths"]["output_midi"],
                output_stem=stem,
                model_size=size,
                device=dev,
                instruments=instr_list,
            )
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

        table = Table(title="Resultado da Transcrição (MuScriptor)", show_header=True)
        table.add_column("Campo", style="cyan")
        table.add_column("Valor", style="green")
        table.add_row("Engine", f"MuScriptor ({result['model_size']})")
        table.add_row("MIDI", result["midi_path"])
        console.print(table)

    # ── Basic Pitch ──────────────────────────────────────────────────────────
    else:
        audio_cfg = settings["audio"]
        paths     = settings["paths"]

        onset_threshold   = onset        if onset        is not None else audio_cfg["onset_threshold"]
        frame_threshold   = frame        if frame        is not None else audio_cfg["frame_threshold"]
        minimum_frequency = min_freq     if min_freq     is not None else audio_cfg["minimum_frequency"]
        maximum_frequency = max_freq     if max_freq     is not None else audio_cfg["maximum_frequency"]
        min_note_length   = min_note_len if min_note_len is not None else audio_cfg["minimum_note_length"]
        melodia           = not no_melodia and audio_cfg["melodia_trick"]

        console.print(Panel.fit(
            f"[bold cyan]Transcrevendo:[/bold cyan] {audio_file.name}  →  [green]{stem}[/green]\n"
            f"engine=[bold]basic-pitch[/bold]  onset={onset_threshold}  frame={frame_threshold}  "
            f"freq=[{minimum_frequency}, {maximum_frequency}] Hz",
            title="music-and-tabs",
        ))

        from modules.audio_processing.transcription import transcribe as run_basic_pitch

        result = run_basic_pitch(
            audio_path=audio_file,
            output_midi_dir=BASE_DIR / paths["output_midi"],
            output_notes_dir=BASE_DIR / paths["output_notes"],
            output_stem=stem,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_length,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            melodia_trick=melodia,
        )

        table = Table(title="Resultado da Transcrição (Basic Pitch)", show_header=True)
        table.add_column("Campo", style="cyan")
        table.add_column("Valor", style="green")
        table.add_row("Engine", "Basic Pitch")
        table.add_row("Notas detectadas", str(result["note_count"]))
        table.add_row("MIDI", result["midi_path"])
        table.add_row("JSON de notas", result["notes_path"])
        console.print(table)


@app.command()
def chords(
    audio_file: Path = typer.Argument(
        ...,
        help="Caminho para o arquivo de áudio (.wav ou .mp3)",
        exists=True,
        readable=True,
    ),
    output_name: str = typer.Option(
        None, "--output-name", "-n",
        help="Nome base do arquivo de saída (sem extensão). Padrão: nome do áudio.",
    ),
    hop: int = typer.Option(None, "--hop", help="Hop length (default 4096)"),
    min_dur: float = typer.Option(None, "--min-dur", help="Duração mínima de segmento em s (default 0.5)"),
) -> None:
    """Detecta acordes no áudio e gera JSON com sequência de acordes."""
    settings = load_settings()
    setup_logging(settings)

    paths = settings["paths"]
    chord_cfg = settings.get("chords", {})

    hop_length = hop if hop is not None else chord_cfg.get("hop_length", 4096)
    min_segment = min_dur if min_dur is not None else chord_cfg.get("min_segment_duration", 0.5)

    stem = output_name if output_name else audio_file.stem

    console.print(Panel.fit(
        f"[bold cyan]Detectando acordes:[/bold cyan] {audio_file.name}  →  [green]{stem}[/green]\n"
        f"hop={hop_length}  min_seg={min_segment}s",
        title="music-and-tabs",
    ))

    from modules.audio_processing.chord_detection import detect_and_export

    result = detect_and_export(
        audio_path=audio_file,
        output_dir=BASE_DIR / "output" / "chords",
        output_stem=stem,
        hop_length=hop_length,
        min_segment_duration=min_segment,
    )

    table = Table(title="Resultado da Detecção de Acordes", show_header=True)
    table.add_column("Campo", style="cyan")
    table.add_column("Valor", style="green")
    table.add_row("Segmentos", str(result["chord_count"]))
    table.add_row("Acordes únicos", ", ".join(result["unique_chords"]))
    table.add_row("JSON", result["chords_path"])
    table.add_row("MIDI", result["midi_path"])
    console.print(table)

    # Prévia da sequência
    preview = Table(title="Sequência de Acordes (primeiros 15)", show_header=True)
    preview.add_column("Início", style="cyan", justify="right")
    preview.add_column("Fim", style="cyan", justify="right")
    preview.add_column("Acorde", style="bold green")
    preview.add_column("Conf.", style="yellow", justify="right")
    for seg in result["segments"][:15]:
        preview.add_row(
            f"{seg['start_time']:.2f}s",
            f"{seg['end_time']:.2f}s",
            seg["chord"],
            f"{seg['confidence']:.2f}",
        )
    console.print(preview)


if __name__ == "__main__":
    app()
