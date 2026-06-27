"""savitr command-line entry point: `savitr ocr ...` and `savitr parse-rolls ...`."""

import sys

USAGE = """savitr — fast Surya OCR on Apple Silicon (electoral rolls)

usage:
  savitr ocr PDF [--terse] [--pages 3-14]          OCR a PDF's pages to voter records
  savitr parse-rolls (-f PDF | -d DIR) -o OUT.csv [--terse]
                                                   parse rolls into the canonical voter CSV

Add --terse to use the distilled terse-Surya model (faster, electoral-roll-specific).
Run `savitr ocr -h` / `savitr parse-rolls -h` for the full options of each subcommand.
"""


def main(argv: list[str] | None = None) -> int:
    """Dispatch `savitr <subcommand>` to the matching entry point."""
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd = argv.pop(1)  # strip subcommand so each main() sees the rest
    sys.argv = argv
    if cmd == "ocr":
        from savitr.rolls.ocr import main as run

        return run()
    if cmd in ("parse-rolls", "parse_rolls", "rolls"):
        from savitr.rolls.pipeline import main as run

        return run()
    print(f"savitr: unknown command {cmd!r}\n", file=sys.stderr)
    print(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
