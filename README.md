<!-- Run this in the native linux mode on the DELL XPS13 laptop under //wsl.localhost/Ubuntu/home/amirchandani/projects/fixPhotoDatesAI/

The repository is on git hub at https://www.github.com/anjuman2/fixPhotoDate

Run it under VSCode with WSL-Ubuntu extension

I am copying the entire fixPhotoDatesAI folder from linux filesystem over to OneDrive/projects/fixPhotoDatesAI just for visibility but DO NOT TRY TO RUN THIS FROM THERE!
>

# Photo Date Recovery Pipeline

Recovers the camera-printed date stamp from ~2,000 scanned 35mm film negatives
and writes the real capture date into each image's EXIF metadata, replacing
the (incorrect) scan-date metadata added by the scanning house.

Scans were produced by a professional scanning house at 6000x4000 (landscape)
or 4000x6000 (portrait) resolution. Many cameras in this collection printed a
small orange/red LED date stamp directly onto the negative in one corner;
others didn't have the feature enabled for a given shot. The pipeline uses
Claude's vision model (via the Anthropic Batch API) to read that stamp, since
OCR tools like Tesseract are unreliable on this LED/dot-matrix font.

## How it works

The date stamp always originates in the same physical spot on the negative,
but depending on how a given scan batch fed the negative into the scanner, it
can end up in any of the 4 corners of the scanned image, each needing a
specific correction (mirror/rotation) to read normally. This mapping differs
between landscape and portrait shots (portrait rotation depends on which way
the camera was turned). See the module docstring in
`datestamp_common_landscape_and_portrait_incl_nodate.py` for the full
reasoning and which corner-corrections are empirically confirmed vs. derived.

For each image, the pipeline crops all 4 corners, pre-corrects each crop's
orientation in code (rather than asking Claude to mentally undo mirroring/
rotation, which proved unreliable), and sends all 4 corner crops to Claude in
one request, asking it to identify which corner (if any) has a genuine stamp
and read its digits.

## Pipeline

Three scripts, run in order:

### 1. `step1_submit_batch.py` — submit
Walks the input folder, builds 4 corner-crop images per photo, and submits
them to Claude via the Anthropic Batch API. Large runs are automatically
split into multiple batches (bounded by size and request count) so no single
batch approaches Anthropic's caps, and so progress is saved to disk after
every chunk rather than only at the end.

```bash
python3 step1_submit_batch.py <input_folder> <work_folder> [--only-orientation portrait|landscape]
```

- `--only-orientation portrait|landscape` — restrict this run to just one
  orientation. Useful for reprocessing a folder against a subset (e.g. after
  fixing portrait-mode handling, rerun only the portrait files into a new
  `work_folder` — `step2` appends to the same output rather than overwriting).

Writes `batch_info.json` in `<work_folder>` tracking every submitted batch.

### 2. `step2_collect_results.py` — collect
Polls each batch from `batch_info.json` until it's done, then for each image:
copies it into the output folder, writes the recovered date into
`DateTimeOriginal`/`DateTimeDigitized` EXIF fields, and logs successes/
failures.

```bash
python3 step2_collect_results.py <work_folder> <output_folder> [--no-wait]
```

- `--no-wait` — check status once; process any batches already finished and
  skip (report) any still in progress. Re-run later to pick up the rest.

Outputs (in `<output_folder>`):
- `processed_dates.csv` — filename + date written
- `needs_review.csv` — filename + failure reason, for anything Claude
  couldn't read or that errored
- `needs_review_crops/` — the 4 corner crops saved for each flagged image, so
  step3 (or you) can see what Claude saw

### 3. `step3_manual_review.py` — manual fallback
Interactive CLI that walks `needs_review.csv` one image at a time, opens its
saved corner crops (auto-opens via Windows Explorer from WSL, with the file
path also printed as a fallback), and prompts for the date.

```bash
python3 step3_manual_review.py <output_folder>
```

At each prompt:
- `MM/DD/YYYY` — write this date to EXIF and remove from the queue
- `s` — skip for now (stays queued for next run)
- `u` — mark permanently unknown/undateable (logged to
  `unresolved_permanently.csv`, no EXIF written — for blank stamps, ruined
  negatives, etc.)
- `q` — quit; progress so far is saved, remaining files stay queued

Safe to interrupt and resume at any time.

## `datestamp_common.py`

Shared utilities used by all three scripts:
- Corner-crop extraction with per-corner/per-orientation transform correction
- The Claude prompt and reply parser (`MM=<n> DD=<n> YY=<n>` or `UNKNOWN`)
- EXIF writing for both JPEG (via `piexif`) and TIFF (via Pillow's own EXIF
  handling — `piexif.insert()` fails silently on TIFF)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install anthropic piexif pillow
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Notes / known limitations

- Portrait-mode corner corrections: TR and BL cases are empirically
  confirmed; TL and BR are derived by symmetry but not yet independently
  confirmed. If portrait `needs_review.csv` entries skew heavily toward
  TL/BR, that mapping is the first thing to check.
- The Claude prompt is intentionally framed as "at most one corner **may**"
  contain a stamp (not "exactly one does") — an earlier, more assertive
  framing caused hallucinated dates on stampless images.
- Date stamps appear in one of two camera-dependent formats
  (`MMDD'YY` or `'YYMMDD`), disambiguated by where the apostrophe falls in
  the string.