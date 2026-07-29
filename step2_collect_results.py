#!/usr/bin/env python3
"""
step2_collect_results.py

Polls every batch submitted by step1 (there may be several now, since large
runs get split into size-bounded chunks) until each is done, then per batch:
    - copies each original image into <output_folder>
    - writes the recovered capture date into EXIF DateTimeOriginal/Digitized
    - logs anything Claude couldn't read (or that errored) to needs_review.csv
      in the output folder, alongside the original crop for a quick manual look

Batches are processed and written to disk one at a time, so if this script
is interrupted partway through a multi-batch run, everything already
finished stays recorded -- re-running just picks up the remaining batches.

Usage:
    python3 step2_collect_results.py <work_folder> <output_folder>

Pass --no-wait if you just want to check status of all batches once: any
that are already done get processed, any still in progress are reported
and skipped (re-run later to pick them up).
"""

import os
import sys
import csv
import json
import time
import shutil
from pathlib import Path

import anthropic

from datestamp_common_landscape_and_portrait_incl_nodate import get_all_corner_crops, parse_claude_date_reply, update_exif

POLL_SECONDS = 60


def save_review_crops(src_path, review_dir):
    """Save all 4 corner candidates so a human can quickly see which one has the stamp."""
    try:
        review_dir.mkdir(exist_ok=True)
        for corner, crop in get_all_corner_crops(src_path):
            crop.save(review_dir / f"{src_path.stem}_{corner}.png")
    except Exception:
        pass


def append_csv_rows(csv_path, header, rows):
    """Append rows to a CSV, writing the header only if the file is new.
    Used for both processed_dates.csv and needs_review.csv so re-running
    step2 (e.g. against later batches) never clobbers earlier records."""
    if not rows:
        return
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerows(rows)


def wait_for_batch(client, batch_id, no_wait):
    """Poll a single batch until it's ended. Returns True if ended, False if
    still in progress and the caller should skip it for now (--no-wait)."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  status={batch.processing_status}  "
              f"succeeded={counts.succeeded} errored={counts.errored} "
              f"processing={counts.processing} canceled={counts.canceled} expired={counts.expired}",
              flush=True)
        if batch.processing_status == "ended":
            return True
        if no_wait:
            print("  Not finished yet -- skipping for now (re-run later to pick it up).", flush=True)
            return False
        time.sleep(POLL_SECONDS)


def process_batch(client, batch_id, custom_id_to_path, output_folder, review_dir):
    """Retrieve and apply results for one already-ended batch. Returns (success, fail) counts."""
    success = fail = 0
    review_rows = []
    processed_rows = []  # (filename, year-month-day) for every successful write

    for entry in client.messages.batches.results(batch_id):
        custom_id = entry.custom_id
        src_path = Path(custom_id_to_path[custom_id])

        dest = output_folder / src_path.name
        counter = 1
        orig_dest = dest
        while dest.exists():
            dest = output_folder / f"{orig_dest.stem}_{counter:03d}{orig_dest.suffix}"
            counter += 1
        shutil.copyfile(src_path, dest)

        if entry.result.type != "succeeded":
            print(f"{src_path.name}: batch request {entry.result.type} -- flagged for review")
            review_rows.append((src_path.name, f"batch_{entry.result.type}"))
            fail += 1
            continue

        reply_text = ""
        for block in entry.result.message.content:
            if getattr(block, "type", None) == "text":
                reply_text += block.text

        parsed = parse_claude_date_reply(reply_text)
        if parsed is None:
            print(f"{src_path.name}: Claude reply not parseable ({reply_text!r}) -- flagged for review")
            review_rows.append((src_path.name, f"unparseable reply: {reply_text!r}"))
            fail += 1
            save_review_crops(src_path, review_dir)
            continue

        year, month, day = parsed
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        if update_exif(dest, year, month, day):
            print(f"{src_path.name}: {date_str} -> EXIF updated")
            processed_rows.append((src_path.name, date_str))
            success += 1
        else:
            review_rows.append((src_path.name, "EXIF write failed"))
            fail += 1

    # Write this batch's results to disk immediately, rather than waiting
    # until every batch in the run has been processed.
    append_csv_rows(output_folder / "processed_dates.csv", ["filename", "date_written"], processed_rows)
    append_csv_rows(output_folder / "needs_review.csv", ["filename", "reason"], review_rows)

    return success, fail, len(review_rows) > 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_wait = "--no-wait" in sys.argv

    if len(args) != 2:
        print("Usage: python3 step2_collect_results.py <work_folder> <output_folder> [--no-wait]")
        sys.exit(1)

    work_folder = Path(args[0]).resolve()
    output_folder = Path(args[1]).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    review_dir = output_folder / "needs_review_crops"

    info_path = work_folder / "batch_info.json"
    if not info_path.exists():
        print(f"Error: {info_path} not found. Did you run step1_submit_batch.py first?")
        sys.exit(1)

    with open(info_path) as f:
        info = json.load(f)

    # Support both the current multi-batch format ({"batches": [...]}) and the
    # older single-batch format, so this script still works against a
    # batch_info.json produced before step1 was updated to chunk large runs.
    if "batches" in info:
        batches = info["batches"]
    else:
        batches = [{"batch_id": info["batch_id"], "custom_id_to_path": info["custom_id_to_path"]}]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set in your environment.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key, timeout=120.0, max_retries=2)

    print(f"Found {len(batches)} batch(es) to process.", flush=True)

    total_success = total_fail = 0
    had_review_items = False
    pending_batch_ids = []

    for i, batch_entry in enumerate(batches, 1):
        batch_id = batch_entry["batch_id"]
        custom_id_to_path = batch_entry["custom_id_to_path"]

        print(f"\n=== Batch {i}/{len(batches)}: {batch_id} ===", flush=True)
        ended = wait_for_batch(client, batch_id, no_wait)
        if not ended:
            pending_batch_ids.append(batch_id)
            continue

        print("Batch finished. Retrieving results...", flush=True)
        success, fail, review_flag = process_batch(client, batch_id, custom_id_to_path, output_folder, review_dir)
        total_success += success
        total_fail += fail
        had_review_items = had_review_items or review_flag
        print(f"  Batch {i}/{len(batches)} done -- updated: {success}, needs review: {fail}", flush=True)

    if had_review_items:
        print(f"\nSome image(s) need manual review -- see {output_folder / 'needs_review.csv'}")
        if review_dir.exists():
            print(f"4 corner-candidate crops for each are saved in {review_dir}/ (suffixed _BR/_BL/_TR/_TL)")

    if pending_batch_ids:
        print(f"\n{len(pending_batch_ids)} batch(es) still in progress, skipped for now:")
        for bid in pending_batch_ids:
            print(f"  {bid}")
        print(f"Re-run:\n  python3 step2_collect_results.py {work_folder} {output_folder}")

    print(f"\nDone. Updated: {total_success}, Needs review: {total_fail}")


if __name__ == "__main__":
    main()