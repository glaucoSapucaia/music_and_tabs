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
    output_name: str = typer.Option(
        None, "--output-name", "-n",
        help="Nome base dos arquivos de saída (sem extensão). Padrão: nome do áudio.",
    ),
    onset: float = typer.Option(None, "--onset", "-o", help="Onset threshold (0-1)"),
    frame: float = typer.Option(None, "--frame", "-f", help="Frame threshold (0-1)"),
    min_freq: float = typer.Option(None, "--min-freq", help="Frequência mínima em Hz"),
    max_freq: float = typer.Option(None, "--max-freq", help="Frequência máxima em Hz"),
    min_note_len: int = typer.Option(None, "--min-note-len", help="Duração mínima de nota em ms"),
    no_melodia: bool = typer.Option(False, "--no-melodia", help="Desativa o melodia trick"),
    melodia: bool = typer.Option(None, "--melodia", help="Ativa o melodia trick (ignora settings)"),
) -> None:
    """Transcreve um arquivo de áudio e gera MIDI + JSON de notas."""
    settings = load_settings()
    setup_logging(settings)

    audio_cfg = settings["audio"]
    paths = settings["paths"]

    # Sobrescreve com valores da CLI se fornecidos
    onset_threshold  = onset        if onset        is not None else audio_cfg["onset_threshold"]
    frame_threshold  = frame        if frame        is not None else audio_cfg["frame_threshold"]
    minimum_frequency = min_freq    if min_freq     is not None else audio_cfg["minimum_frequency"]
    maximum_frequency = max_freq    if max_freq     is not None else audio_cfg["maximum_frequency"]
    min_note_length   = min_note_len if min_note_len is not None else audio_cfg["minimum_note_length"]
    melodia_flag = audio_cfg["melodia_trick"]
    if melodia is not None:
        melodia_flag = melodia
    elif no_melodia:
        melodia_flag = False

    stem = output_name if output_name else audio_file.stem

    console.print(Panel.fit(
        f"[bold cyan]Transcrevendo:[/bold cyan] {audio_file.name}  →  [green]{stem}[/green]\n"
        f"onset={onset_threshold}  frame={frame_threshold}  "
        f"freq=[{minimum_frequency}, {maximum_frequency}] Hz  "
        f"min_note={min_note_length}ms  melodia={melodia}",
        title="music-and-tabs",
    ))

    from modules.audio_processing.transcription import transcribe as run_transcription

    result = run_transcription(
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

    table = Table(title="Resultado da Transcrição", show_header=True)
    table.add_column("Campo", style="cyan")
    table.add_column("Valor", style="green")
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


@app.command()
def process(
    audio_file: Path = typer.Argument(
        ...,
        help="Caminho para o arquivo de áudio (.wav)",
        exists=True,
        readable=True,
    ),
    output_name: str = typer.Option(
        None, "--output-name", "-n",
        help="Nome base das saídas (padrão: nome do áudio + versão)",
    ),
    version: str = typer.Option("v4", "--version", "-v", help="Rótulo da versão (v4, v5...)"),
    lowcut: float = typer.Option(75.0, "--lowcut", help="Corte grave do filtro (Hz)"),
    highcut: float = typer.Option(1500.0, "--highcut", help="Corte agudo do filtro (Hz)"),
    onset_delta: float = typer.Option(0.09, "--delta", help="Sensibilidade do onset detector"),
    onset_wait: int = typer.Option(10, "--wait", help="Frames mínimos entre onsets (menor = detecta strums)"),
    no_noise: bool = typer.Option(False, "--no-noise", help="Pula redução de ruído"),
    no_preprocess: bool = typer.Option(False, "--no-preprocess", help="Pula todo o pré-processamento (usa o WAV bruto)"),
    onset_only: bool = typer.Option(False, "--onset-only", help="Modo tablatura: só ataques, duração fixa, onsets próximos viram acorde"),
    chord_cluster: float = typer.Option(0.060, "--chord-cluster", help="Janela de clustering de acorde em segundos"),
    no_quantize: bool = typer.Option(False, "--no-quantize", help="Pula quantização rítmica"),
    max_poly: int = typer.Option(6, "--max-poly", help="Máximo de notas simultâneas"),
    min_dur: float = typer.Option(80.0, "--min-dur", help="Duração mínima de nota (ms)"),
    tempo: float = typer.Option(120.0, "--tempo", help="BPM do MIDI"),
) -> None:
    """Pipeline completo: pré-processa, detecta onsets, transcreve, corrige e exporta MIDI + métricas."""
    settings = load_settings()
    setup_logging(settings)

    stem = output_name if output_name else f"{audio_file.stem}_{version}"

    from modules.audio_processing.pipeline import run_pipeline

    with console.status("[bold cyan]Executando pipeline...[/bold cyan]") as status:
        status.update("Pré-processamento + onsets + transcrição por janela + pós-processamento")
        result = run_pipeline(
            audio_path=audio_file,
            output_dir=BASE_DIR / "output" / "processed",
            output_stem=stem,
            version_label=version,
            skip_preprocess=no_preprocess,
            onset_only=onset_only,
            chord_cluster_s=chord_cluster,
            lowcut=lowcut,
            highcut=highcut,
            reduce_noise=not no_noise,
            onset_delta=onset_delta,
            onset_wait=onset_wait,
            quantize_rhythm=not no_quantize,
            min_note_duration_ms=min_dur,
            max_polyphony=max_poly,
            tempo=tempo,
        )

    m = result["metrics"]

    table = Table(title=f"Resultado do Pipeline — {version}", show_header=True)
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green")
    table.add_row("Notas (saída)", str(m.get("note_count", "?")))
    table.add_row("Onsets detectados", str(m.get("onset_count", "?")))
    table.add_row("Cobertura de onsets", f"{m.get('onset_coverage', 0)*100:.1f}%")
    table.add_row("Polifonia máxima", str(m.get("max_polyphony", "?")))
    table.add_row("Densidade", f"{m.get('density_notes_per_sec', '?')} notas/s")
    table.add_row("Pitch range", f"{m.get('pitch_min', '?')}–{m.get('pitch_max', '?')}")
    table.add_row("Notas < 100ms", str(m.get("notes_under_100ms", "?")))
    table.add_row("BPM estimado", str(m.get("estimated_bpm", "?")))
    table.add_row("SNR do áudio", f"{m.get('snr_estimate_db', '?')} dB")
    table.add_row("MIDI", result["midi_path"])
    table.add_row("Métricas", result["metrics_path"])
    console.print(table)

    # Estatísticas de filtragem do pós-processamento
    stats = m.get("postprocess_stats", {})
    if stats:
        ftable = Table(title="Filtragem (pós-processamento)")
        ftable.add_column("Etapa", style="cyan")
        ftable.add_column("Notas restantes", style="green", justify="right")
        ftable.add_row("Entrada (brutas)", str(stats.get("input", "?")))
        ftable.add_row("Após range do violão", str(stats.get("after_range_filter", "?")))
        ftable.add_row("Após curtas removidas", str(stats.get("after_short_filter", "?")))
        ftable.add_row("Após deduplicação", str(stats.get("after_deduplication", "?")))
        ftable.add_row("Após limite de polifonia", str(stats.get("after_polyphony_limit", "?")))
        console.print(ftable)


if __name__ == "__main__":
    app()
