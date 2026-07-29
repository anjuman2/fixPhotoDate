#!/usr/bin/env python3
"""
step3_manual_review.py

Walks needs_review.csv (produced by step2) one file at a time, opens its 4
saved corner-candidate crops so you can see them, and prompts you to type in
the date. Writes EXIF immediately and removes the file from the review queue
as you go -- safe to quit (Ctrl+C or 'q') and resume later; already-resolved
files won't be asked again.

Usage:
    python3 step3_manual_review.py <output_folder>

At each prompt you can enter:
    - a date as  MM/DD/YYYY   (e.g. 10/19/1996)
    - "s"  to skip this one for now (stays in the queue for next time)
    - "u"  to mark it as genuinely unknown/undateable (removed from the
           queue permanently, logged to unresolved_permanently.csv, no EXIF
           written -- use this for e.g. blank stamps, ruined negatives)
    - "q"  to quit; progress so far is saved, remaining files stay queued
"""

import os
import sys
import csv
import shutil
import subprocess
from pathlib import Path

from datestamp_common_landscape_and_portrait_incl_nodate import update_exif


def try_open_in_windows(path: Path):
    """
    Best-effort: open the crop in whatever Windows uses for images (Photos,
    etc.) via explorer.exe, which WSL can call directly. This often returns a
    non-zero exit code even on success (a known WSL quirk), so we don't treat
    failure as fatal -- the filename is always printed too, as a fallback.
    """
    try:
        wpath = subprocess.run(
            ["wslpath", "-w", str(path)], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if wpath:
            subprocess.Popen(["explorer.exe", wpath],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except Exception:
        pass
    return False


def parse_date_input(s):
    s = s.strip()
    for sep in ("/", "-"):
        if sep in s:
            parts = s.split(sep)
            if len(parts) == 3:
                try:
                    month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                    if year < 100:
                        year = 1900 + year if year >= 50 else 2000 + year
                    if 1 <= month <= 12 and 1 <= day <= 31 and 1800 <= year <= 2100:
                        return (year, month, day)
                except ValueError:
                    pass
    return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 step3_manual_review.py <output_folder>")
        sys.exit(1)

    output_folder = Path(sys.argv[1]).resolve()
    review_csv = output_folder / "needs_review.csv"
    crops_dir = output_folder / "needs_review_crops"
    permanently_unresolved_csv = output_folder / "unresolved_permanently.csv"

    if not review_csv.exists():
        print(f"No {review_csv} found -- nothing to review.")
        sys.exit(0)

    with open(review_csv, newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]

    if not rows:
        print("needs_review.csv is empty -- nothing to review.")
        sys.exit(0)

    print(f"{len(rows)} file(s) to review. For each: MM/DD/YYYY to set the date, "
          "'s' to skip for now, 'u' for permanently unknown, 'q' to quit.\n")

    remaining = []
    permanently_unresolved = []
    resolved_count = 0

    quit_early = False
    for idx, row in enumerate(rows, 1):
        filename, reason = row[0], row[1] if len(row) > 1 else ""
        if quit_early:
            remaining.append(row)
            continue

        image_path = output_folder / filename
        stem = Path(filename).stem
        crop_files = sorted(crops_dir.glob(f"{stem}_*.png")) if crops_dir.exists() else []

        print(f"[{idx}/{len(rows)}] {filename}")
        print(f"    reason logged: {reason}")
        if not image_path.exists():
            print(f"    WARNING: {image_path} not found -- can't write EXIF for this one.")
        if crop_files:
            print(f"    opening {len(crop_files)} corner crop(s)...")
            for cf in crop_files:
                opened = try_open_in_windows(cf)
                if not opened:
                    print(f"      (couldn't auto-open -- view manually: {cf})")
        else:
            print(f"    no saved crops found -- open the original manually: {image_path}")

        while True:
            answer = input("    Date (MM/DD/YYYY) / s=skip / u=unknown / q=quit: ").strip()
            if answer.lower() == "q":
                quit_early = True
                remaining.append(row)
                break
            if answer.lower() == "s":
                remaining.append(row)
                break
            if answer.lower() == "u":
                permanently_unresolved.append(row)
                break
            parsed = parse_date_input(answer)
            if parsed is None:
                print("    Couldn't parse that -- use MM/DD/YYYY, or s/u/q.")
                continue
            year, month, day = parsed
            if not image_path.exists():
                print(f"    Can't write EXIF, file missing. Treating as skipped.")
                remaining.append(row)
                break
            if update_exif(image_path, year, month, day):
                print(f"    -> EXIF set to {year:04d}-{month:02d}-{day:02d}")
                resolved_count += 1
            else:
                remaining.append(row)
            break
        print()

    # Rewrite needs_review.csv with only what's still outstanding
    with open(review_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(remaining)

    if permanently_unresolved:
        write_header = not permanently_unresolved_csv.exists()
        with open(permanently_unresolved_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerows(permanently_unresolved)

    print(f"Done this session. Resolved: {resolved_count}, "
          f"marked permanently unknown: {len(permanently_unresolved)}, "
          f"still queued: {len(remaining)}")
    if remaining:
        print(f"Re-run this script anytime to keep going -- {review_csv} still has {len(remaining)} left.")


if __name__ == "__main__":
    main()
