#!/usr/bin/env python3
"""
step1_submit_batch.py

Walks the input folder, locates the date-stamp corner in every image, and
submits Batch API requests (4 corner-crop images each) to Claude Haiku 4.5
asking it to read the date. Doesn't wait for results -- that's step2's job
(batch jobs can take up to a few hours).

Large runs are split into multiple batches automatically, so no single
batch gets anywhere near Anthropic's 256 MB / 100,000-request cap, and so
that a crash partway through doesn't lose work already submitted. Each
chunk is submitted and recorded to disk as soon as it's built -- images
aren't all held in memory at once.

Usage:
    python3 step1_submit_batch.py <input_folder> <work_folder>

<work_folder> is where this script stashes:
    - batch_info.json   ({"model": ..., "batches": [{"batch_id", "custom_id_to_path"}, ...]})
                         Written/updated after EVERY chunk, not just at the end.

Requires: ANTHROPIC_API_KEY set in your environment.
"""

import os
import sys
import gc
import json
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from datestamp_common import get_all_corner_crops, crop_to_base64_jpeg, CLAUDE_PROMPT, CORNERS

MODEL = "claude-haiku-4-5-20251001"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Anthropic's batch cap is 256 MB / 100,000 requests. We stay well under
# that on purpose: partly for safety margin, partly because building and
# serializing a payload anywhere near the real cap is itself a memory
# spike we'd rather not hit on top of whatever the image-processing loop
# has already accumulated.
CHUNK_MAX_BYTES = 40 * 1024 * 1024   # 40 MB of base64 image data per batch
CHUNK_MAX_REQUESTS = 300            # ...or this many requests, whichever first


def find_images(folder):
    files = []
    for ext in VALID_EXTS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def build_request_content(image_path):
    """4 image blocks (one per corner, in fixed BR/BL/TR/TL order) + the prompt.
    Returns (content, byte_size) where byte_size is the total base64 payload size,
    so the caller can track chunk size without re-measuring later."""
    content = []
    total_bytes = 0
    for corner, crop in get_all_corner_crops(image_path):
        b64 = crop_to_base64_jpeg(crop)
        total_bytes += len(b64)
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            }
        )
    content.append({"type": "text", "text": CLAUDE_PROMPT})
    return content, total_bytes


def submit_chunk(client, chunk_requests, chunk_custom_id_to_path, chunk_bytes, work_folder, batches_so_far):
    """Submit one chunk as its own batch, then immediately persist progress to disk."""
    n = len(chunk_requests)
    mb = chunk_bytes / (1024 * 1024)
    print(f"\nSubmitting chunk: {n} request(s), ~{mb:.1f} MB of image data...", flush=True)

    batch = client.messages.batches.create(requests=chunk_requests)
    print(f"  -> Batch created: {batch.id}  (status: {batch.processing_status})", flush=True)

    batches_so_far.append(
        {
            "batch_id": batch.id,
            "custom_id_to_path": chunk_custom_id_to_path,
        }
    )

    # Write the whole accumulated list back out after every chunk. This is the
    # key safety net: if a later chunk hangs or the process dies, everything
    # submitted so far is already safely recorded, not lost.
    with open(work_folder / "batch_info.json", "w") as f:
        json.dump({"model": MODEL, "batches": batches_so_far}, f, indent=2)
    print(f"  -> Progress saved to {work_folder / 'batch_info.json'} "
          f"({len(batches_so_far)} batch(es) so far)", flush=True)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 step1_submit_batch.py <input_folder> <work_folder>")
        sys.exit(1)

    input_folder = Path(sys.argv[1]).resolve()
    work_folder = Path(sys.argv[2]).resolve()
    work_folder.mkdir(parents=True, exist_ok=True)

    if not input_folder.exists():
        print(f"Error: input folder does not exist: {input_folder}")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set in your environment.")
        print('Run:  export ANTHROPIC_API_KEY="sk-ant-..."   then try again.')
        sys.exit(1)

    # Explicit timeout so a stuck network call surfaces as a clear error
    # after a bounded wait instead of hanging indefinitely.
    client = anthropic.Anthropic(api_key=api_key, timeout=120.0, max_retries=2)

    images = find_images(input_folder)
    print(f"Found {len(images)} image(s) in {input_folder}", flush=True)
    print(f"Submitting in chunks of up to {CHUNK_MAX_REQUESTS} requests "
          f"or {CHUNK_MAX_BYTES / (1024*1024):.0f} MB, whichever comes first.\n", flush=True)

    batches_so_far = []
    load_errors = []

    chunk_requests = []
    chunk_custom_id_to_path = {}
    chunk_bytes = 0

    total_queued = 0

    for i, img_path in enumerate(images, 1):
        try:
            content, content_bytes = build_request_content(img_path)
        except Exception as e:
            print(f"[{i}/{len(images)}] {img_path.name}: ERROR building crops ({e}) -- skipped", flush=True)
            load_errors.append((img_path.name, str(e)))
            continue

        # If adding this image would push the current chunk over either limit,
        # submit what we have first, then start a fresh chunk.
        if chunk_requests and (
            chunk_bytes + content_bytes > CHUNK_MAX_BYTES
            or len(chunk_requests) >= CHUNK_MAX_REQUESTS
        ):
            submit_chunk(client, chunk_requests, chunk_custom_id_to_path, chunk_bytes, work_folder, batches_so_far)
            chunk_requests = []
            chunk_custom_id_to_path = {}
            chunk_bytes = 0
            gc.collect()

        custom_id = f"img-{i:05d}"
        chunk_requests.append(
            Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=20,
                    messages=[{"role": "user", "content": content}],
                ),
            )
        )
        chunk_custom_id_to_path[custom_id] = str(img_path)
        chunk_bytes += content_bytes
        total_queued += 1
        print(f"[{i}/{len(images)}] {img_path.name}: queued (4 corner candidates, "
              f"chunk {len(chunk_requests)}/{CHUNK_MAX_REQUESTS}, "
              f"~{chunk_bytes/(1024*1024):.1f} MB)", flush=True)

    # Submit whatever's left in the final partial chunk.
    if chunk_requests:
        submit_chunk(client, chunk_requests, chunk_custom_id_to_path, chunk_bytes, work_folder, batches_so_far)

    if load_errors:
        print(f"\n{len(load_errors)} image(s) failed to process and were skipped:", flush=True)
        for name, err in load_errors:
            print(f"  {name}: {err}", flush=True)

    if not batches_so_far:
        print("Nothing was submitted.", flush=True)
        sys.exit(0)

    print(f"\nDone. {total_queued} image(s) queued across {len(batches_so_far)} batch(es).", flush=True)
    print("Batches typically finish within an hour, sometimes a few hours for larger jobs.", flush=True)
    print(f"\nWhen ready, run:\n  python3 step2_collect_results.py {work_folder} <output_folder>", flush=True)


if __name__ == "__main__":
    main()