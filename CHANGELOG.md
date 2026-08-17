# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] - 2026-08-17

### Added

- Pinned Hugging Face model resolution and parser regression tests.
- Reusable py-canon CI, docs, Dependabot, and trusted-publishing workflows.

### Changed

- Migrated packaging to `uv_build` with a `src/` layout.
- Made the README the documentation landing page while retaining autodoc.
- Preserved serial numbers in terse round trips and made deduplication prefer
  stable EPIC and serial identifiers.
- Required a patched Pillow release and raised the CI coverage floor to 30%.

### Removed

- Removed the optional monkey-patched Surya backend and its vulnerable
  `surya-ocr` dependency. `MLXSuryaOCR` remains the supported runtime.

## [0.2.0] - 2026-07-18

[Unreleased]: https://github.com/in-rolls/savitr/commits/main
