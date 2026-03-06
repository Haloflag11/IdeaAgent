"""Banner utilities for IdeaAgent CLI startup display."""

# ── ASCII art logo ────────────────────────────────────────────────────────────
#
# Font: Big (pyfiglet-style, pure ASCII, terminal-safe)
#

IDEA_AGENT_ASCII = r"""
 ___ ____  _____    _         _    ____ _____ _   _ _____
|_ _|  _ \| ____|  / \       / \  / ___| ____| \ | |_   _|
 | || | | |  _|   / _ \     / _ \| |  _|  _| |  \| | | |
 | || |_| | |___ / ___ \   / ___ \ |_| | |___| |\  | | |
|___|____/|_____/_/   \_\ /_/   \_\____|_____|_| \_| |_|
"""

TAGLINE = "✦  Experimental Agent for ML Research  ✦"
VERSION = "v0.1.0"

SHORTCUT_HINT = "Type [bold bright_cyan]?[/bold bright_cyan] [dim]or[/dim] [bold bright_cyan]/help[/bold bright_cyan] [dim]for commands[/dim]"


def get_banner_text():
    """Return a Rich :class:`~rich.text.Text` object for the startup banner.

    The text uses a purple/violet gradient palette so it looks vivid even on
    dark and light terminals.  The surrounding :class:`~rich.panel.Panel` is
    drawn by the caller so that layout (padding, title, etc.) stays in *cli.py*.

    Returns:
        rich.text.Text: Styled banner text ready to be passed to ``console.print``.
    """
    from rich.text import Text

    t = Text(justify="center")

    # ── ASCII logo lines with per-line colour shifts ──────────────────────
    logo_lines = IDEA_AGENT_ASCII.strip("\n").splitlines()

    # Colour palette: purple → violet → magenta gradient
    colours = [
        "bright_magenta",   # row 0
        "#cc66ff",          # row 1
        "#b84dff",          # row 2
        "#a333ff",          # row 3
        "#9933ff",          # row 4
        "#8000ff",          # row 5
    ]

    for i, line in enumerate(logo_lines):
        colour = colours[i % len(colours)]
        t.append(line + "\n", style=f"bold {colour}")

    # ── Tagline ───────────────────────────────────────────────────────────
    t.append("\n")
    t.append(TAGLINE, style="bold #cc66ff")
    t.append("   ")
    t.append(VERSION, style="dim #9933ff")
    t.append("\n")

    return t


def get_banner_panel():
    """Return a Rich :class:`~rich.panel.Panel` wrapping the ASCII banner.

    The panel uses a vivid purple border to give the "cool purple box" effect
    requested in the task.  It is self-contained – just pass the return value
    directly to ``console.print()``.

    Returns:
        rich.panel.Panel: Panel object ready for printing.
    """
    from rich.panel import Panel

    banner_text = get_banner_text()

    return Panel(
        banner_text,
        border_style="bold #9933ff",   # vivid purple border
        padding=(0, 2),
        subtitle=SHORTCUT_HINT,
        subtitle_align="center",
    )
