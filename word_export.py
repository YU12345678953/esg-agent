from pathlib import Path
from typing import Any

import pypandoc

try:
    from .markdown_sanitizer import sanitize_markdown_for_pypandoc
except ImportError:
    from markdown_sanitizer import sanitize_markdown_for_pypandoc


def convert_markdown_to_word(
    input_file: str | Path,
    output_file: str | Path,
    *,
    sanitized_markdown_file: str | Path | None = None,
    extra_args: list[str] | None = None,
    **pandoc_kwargs: Any,
) -> Path:
    """
    Standardize Markdown first, then convert it to Word with pypandoc.

    If sanitized_markdown_file is not provided, a sibling file named
    "<input>.pandoc.md" is created for easier debugging.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    sanitized_path = (
        Path(sanitized_markdown_file)
        if sanitized_markdown_file is not None
        else input_path.with_suffix(".pandoc.md")
    )

    markdown_text = input_path.read_text(encoding="utf-8")
    sanitized_text = sanitize_markdown_for_pypandoc(markdown_text)
    sanitized_path.write_text(sanitized_text, encoding="utf-8")

    output = pypandoc.convert_file(
        str(sanitized_path),
        "docx",
        outputfile=str(output_path),
        extra_args=extra_args or [],
        **pandoc_kwargs,
    )
    if output:
        raise RuntimeError(f"Error converting file: {output}")

    return output_path
