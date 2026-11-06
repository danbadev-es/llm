"""Download Don Quijote (Project Gutenberg #2000) and save it as plain text."""

import urllib.request
from pathlib import Path

SOURCE_URL = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "quijote" / "input.txt"


def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep only the book body between Gutenberg's START and END markers."""
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start == -1 or end == -1:
        raise ValueError("Gutenberg markers not found; the source format may have changed.")
    body_start = text.index("\n", start) + 1
    return text[body_start:end].strip() + "\n"


def main() -> None:
    if OUTPUT_PATH.exists():
        print(f"Already exists: {OUTPUT_PATH}")
        return

    print(f"Downloading {SOURCE_URL} ...")
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        raw = response.read().decode("utf-8-sig")

    text = strip_gutenberg_boilerplate(raw.replace("\r\n", "\n"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"Saved {len(text):,} characters to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
