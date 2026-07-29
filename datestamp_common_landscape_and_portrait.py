#!/usr/bin/env python3
"""
datestamp_common_landscape_and_portrait.py

Locates which corner of a scanned photo the orange/red LED date stamp is
in. The physical stamp always originates in the same spot on the negative
(bottom-right, reading normally). Depending on how a given scan batch fed
the negative/print into the scanner, that content can show up in any of
the other 3 corners with a specific, predictable transform needed to
correct it back to normal reading:

    - BR (identity):       normal, no correction needed
    - BL (horizontal mirror):   date reads mirrored left-right
    - TR (vertical flip):       date reads upside-down, left-right order preserved
    - TL (180 degree rotation): date reads upside-down AND mirrored (both flips combined)

This is the Klein four-group of flip transforms (no 90/270 rotations,
just identity + the 2 single-axis flips + the double-flip). BR<->TL are
diagonal opposites needing the full rotation; BR<->BL and BR<->TR are
edge-adjacent needing a single-axis flip each. The TL=180-rotation case
was confirmed empirically against real ground-truth dates -- a single
vertical flip alone reproduced a different (wrong) digit segmentation,
while the full rotation exactly reproduced the confirmed-correct one.

This mapping is specific to this scan job: a single professional scanning
house ran all 1971 images through one consistent process, all at 6000x4000
(landscape) or 4000x6000 (portrait) resolution. (An earlier one-off test
image from an unrelated, informally-scanned batch -- Negative0028 -- had
its stamp reading correctly with no correction, contradicting this
mapping, but that image didn't go through the same process as this job's
1971 files and isn't representative of it.)

PORTRAIT MODE (source image is 4000x6000, i.e. taller than wide):

The film always feeds through the scanner along its toothed sprocket edge,
which is a fixed physical axis independent of how the camera was held. So
when the camera was rotated 90 degrees to shoot a portrait photo, the
whole frame -- including the stamp -- rotates relative to that fixed edge.
There are 4 portrait cases, mirroring the landscape structure:

    - TR (rotate 90 deg clockwise to read):    camera turned counter-clockwise
    - BL (rotate 90 deg counter-clockwise):     camera turned clockwise
    - TL: mirror of TR (negative scanned inside-out)
    - BR: mirror of BL (negative scanned inside-out)

TR and BL are confirmed against real examples from this job. TL and BR are
derived by the same mirror-pairing logic used for the landscape corners,
but not yet independently confirmed -- if portrait needs_review looks
unexpectedly high for TL/BR specifically, that's the first thing to check.

Earlier approaches that proved too brittle for this:
  1. Pixel-color localization (find orange pixels, cluster into a bounding
     box) breaks down whenever the background near the stamp is also
     warm-toned (wood furniture, skin, etc. -- confirmed on real photos
     where 70% of an entire corner crop was "orange" by color threshold).
  2. Asking Claude to mentally correct mirroring/rotation itself, rather
     than pre-correcting in code, turned out to be unreliable specifically
     for the 180-degree-rotation case -- confirmed on a real batch where
     the same physical stamp produced a different wrong answer almost
     every run, including plausible-looking wrong dates that passed
     validation. Pre-correcting removes that failure mode.
"""

import io
import re
import base64
from pathlib import Path

import piexif
from PIL import Image

# Fraction of the image's width/height used for each corner candidate crop.
CROP_W_FRAC = 0.32
CROP_H_FRAC = 0.24

CORNERS = ["BR", "BL", "TR", "TL"]
CORNER_LABELS = {
    "BR": "A (bottom-right)",
    "BL": "B (bottom-left)",
    "TR": "C (top-right)",
    "TL": "D (top-left)",
}

# Default per-corner correction transform. See module docstring for the
# reasoning and the important caveat about this not being universal across
# every scan batch.
CORNER_TRANSFORMS = {
    "BR": lambda im: im,
    "BL": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "TR": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
    "TL": lambda im: im.rotate(180),
}

# Portrait-mode equivalent (source image taller than wide). See module
# docstring for the sprocket-edge reasoning and the confirmed-vs-derived
# status of each corner. Image.ROTATE_270 rotates 90 deg clockwise;
# Image.ROTATE_90 rotates 90 deg counter-clockwise (PIL's convention).
PORTRAIT_CORNER_TRANSFORMS = {
    "TR": lambda im: im.transpose(Image.ROTATE_270),
    "BL": lambda im: im.transpose(Image.ROTATE_90),
    "TL": lambda im: im.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT),
    "BR": lambda im: im.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT),
}


def _get_corner_crop(im, corner, cw, ch):
    w, h = im.size
    if corner == "BR":
        box = (w - cw, h - ch, w, h)
    elif corner == "BL":
        box = (0, h - ch, cw, h)
    elif corner == "TR":
        box = (w - cw, 0, w, ch)
    elif corner == "TL":
        box = (0, 0, cw, ch)
    else:
        raise ValueError(corner)
    return im.crop(box)


def get_all_corner_crops(image_path, pre_rotate=0, correct_orientation=True):
    """
    Returns an ordered list of (corner_code, PIL.Image) for all 4 corners.

    correct_orientation (default True): apply CORNER_TRANSFORMS to each
    corner crop so Claude receives already right-reading text regardless
    of which corner the stamp turns out to be in. Set False to get raw,
    unmodified crops instead (the older behavior, relying on Claude to
    mentally correct mirroring/rotation -- kept available since it may
    suit batches where CORNER_TRANSFORMS' assumptions don't hold).

    pre_rotate: degrees (0/90/180/270) to rotate the WHOLE source image
    before cropping corners AND before the per-corner correction above.
    Useful for the rarer case where an entire batch needs a rotation that
    doesn't fit the standard 4-corner model at all.
    """
    im = Image.open(image_path).convert("RGB")
    if pre_rotate:
        im = im.rotate(-pre_rotate, expand=True)
    w, h = im.size
    is_portrait = h > w

    if is_portrait:
        # The physical stamp is a fixed size relative to the frame; in
        # portrait mode the frame's long/short axes are swapped relative
        # to landscape, so the crop fractions need to swap too (narrow
        # across the now-short width, tall across the now-long height).
        cw, ch = int(w * CROP_H_FRAC), int(h * CROP_W_FRAC)
        transforms = PORTRAIT_CORNER_TRANSFORMS
    else:
        cw, ch = int(w * CROP_W_FRAC), int(h * CROP_H_FRAC)
        transforms = CORNER_TRANSFORMS

    crops = []
    for corner in CORNERS:
        crop = _get_corner_crop(im, corner, cw, ch)
        if correct_orientation:
            crop = transforms[corner](crop)
        crops.append((corner, crop))
    return crops


def get_image_orientation(image_path):
    """Cheap orientation check (portrait/landscape) without decoding pixel
    data -- just opens the file and reads its header-reported size. Used by
    step1 to filter a rerun down to only the images that need it."""
    with Image.open(image_path) as im:
        w, h = im.size
    return "portrait" if h > w else "landscape"


def crop_to_base64_jpeg(crop_img, target_max_dim=1300):
    """
    Normalize every crop to roughly the same size before sending to Claude,
    regardless of the source scan's resolution. Small crops (from small scans)
    get upscaled so fine LED segments are visible; large crops (from high-res
    scans) get downscaled so we're not paying to ship a multi-megabyte image
    for a tiny text stamp. Without this, a 6000x4000 source image produces a
    corner crop alone bigger than several of the small scans combined.

    Also applies a mild contrast + saturation boost. Some scans have the
    orange/red stamp sitting against a background that's uncomfortably close
    in brightness (e.g. light green grass) -- confirmed on a real photo where
    even our own diagnostic color threshold needed loosening substantially to
    separate the stamp from the background at all. A modest, generic boost
    helps those borderline cases stand out without being aggressive enough to
    introduce color artifacts on the clearer, higher-contrast crops.
    """
    from PIL import ImageEnhance

    longest_side = max(crop_img.width, crop_img.height)
    scale = target_max_dim / longest_side
    new_size = (max(1, round(crop_img.width * scale)), max(1, round(crop_img.height * scale)))
    resized = crop_img.resize(new_size, Image.LANCZOS).convert("RGB")

    resized = ImageEnhance.Contrast(resized).enhance(1.3)
    resized = ImageEnhance.Color(resized).enhance(1.3)

    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def update_exif(image_path, year, month, day):
    """
    Write DateTimeOriginal/DateTimeDigitized into the image's EXIF, in
    whichever way that image format actually supports.

    JPEG goes through piexif, which handles a lot of quirky real-world EXIF
    edge cases well. TIFF goes through Pillow's own Exif object instead --
    piexif.insert() only supports JPEG under the hood and fails silently on
    TIFF with a blank-message exception (InvalidImageDataError with no
    text), and piexif's own loader can also choke reading certain TIFF-
    specific tags (e.g. some scanner MakerNote blobs) that Pillow reads
    without issue. Confirmed against a real Plustek/SilverFast TIFF scan.
    """
    image_path = Path(image_path)
    date_str = f"{year:04d}:{month:02d}:{day:02d} 12:00:00"
    suffix = image_path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        try:
            im = Image.open(image_path)
            exif = im.getexif()
            EXIF_IFD_TAG = 0x8769
            DATETIME_ORIGINAL = 0x9003
            DATETIME_DIGITIZED = 0x9004
            exif_ifd = exif.get_ifd(EXIF_IFD_TAG)
            exif_ifd[DATETIME_ORIGINAL] = date_str
            exif_ifd[DATETIME_DIGITIZED] = date_str
            exif[EXIF_IFD_TAG] = exif_ifd
            im.save(image_path, format="TIFF", exif=exif)
            return True
        except Exception as e:
            print(f"    EXIF write failed (TIFF): {e!r}")
            return False

    try:
        try:
            exif_dict = piexif.load(str(image_path))
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = date_str.encode("utf-8")
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = date_str.encode("utf-8")
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(image_path))
        return True
    except Exception as e:
        print(f"    EXIF write failed: {e!r}")
        return False


def parse_claude_date_reply(text):
    """
    Expects Claude to reply with exactly 'MM=<n> DD=<n> YY=<n>' (in any
    order -- we look up by label, not position) or 'UNKNOWN'. Returns
    (year, month, day) or None.
    """
    text = (text or "").strip()
    if not text or text.upper() == "UNKNOWN":
        return None

    values = {}
    for key in ("MM", "DD", "YY"):
        m = re.search(rf"{key}\s*=\s*(\d{{1,2}})", text, re.IGNORECASE)
        if not m:
            return None
        values[key] = int(m.group(1))

    month, day, yy = values["MM"], values["DD"], values["YY"]
    if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= yy <= 99):
        return None
    year = 1900 + yy if yy >= 50 else 2000 + yy
    return (year, month, day)


CLAUDE_PROMPT = (
    "These are the four corners of a scanned 35mm film photo, labeled A "
    "(bottom-right), B (bottom-left), C (top-right), and D (top-left). "
    "Each has already been corrected for any mirroring or rotation from "
    "scanning, so if a date stamp is present it should read normally, "
    "left-to-right, right-side up. Exactly one of these four corners "
    "contains a small orange or red digital date stamp printed by the "
    "camera. The other three corners will just show background (wall, "
    "furniture, wood grain, etc.) with no date stamp -- ignore any "
    "warm/orange-toned wood, skin tone, or similar background color, and "
    "look specifically for a stamp made of small blocky LED/LCD-style "
    "digit segments.\n\n"
    "(In the rare case a corner's correction doesn't quite match how that "
    "particular photo was actually scanned, the stamp might still appear "
    "mirrored and/or upside-down -- if so, mentally correct for that the "
    "same way you'd read a flipped clock face, and report the true digit "
    "sequence as printed, not as it visually appears.)\n\n"
    "The stamp will be in ONE of two possible formats -- different cameras "
    "in this collection used different formats, so check which one you're "
    "looking at before reading the digits:\n\n"
    "FORMAT 1 -- MMDD'YY (apostrophe appears AFTER some digits, near the "
    "end, just before the last two digits): month, then day, then an "
    "apostrophe, then a 2-digit year. BOTH month and day may be ONE or TWO "
    "digits (neither is zero-padded when it's a single digit) -- e.g. "
    "\"10 9'00\" is October 9, 2000; \"1019'96\" is October 19, 1996; and "
    "\"6 19'03\" (single-digit month) is June 19, 2003. Count the "
    "characters carefully before deciding where month ends and day "
    "begins -- do not assume the month is always 2 digits.\n\n"
    "FORMAT 2 -- 'YYMMDD (apostrophe appears FIRST, before any digits): "
    "a 2-digit year immediately follows the apostrophe, then month, then "
    "day. IMPORTANT: just like Format 1, the year/month/day groups in "
    "this format are often printed with visible gaps between them (e.g. "
    "\"'98 10 3\") -- a gap does NOT change the order. Even when it looks "
    "like three separate groups, if the FIRST group has the apostrophe, "
    "that first group is always the YEAR, not the month -- do not fall "
    "back to Format 1's month-first order just because the groups are "
    "visually separated. So \"'98 10 3\" is YY=98 MM=10 DD=3, i.e. "
    "October 3, 1998 (NOT month=98, which isn't even a valid month -- if "
    "your reading of any format gives a month over 12 or a day over 31, "
    "that's a sign you have the field order backwards and should re-check "
    "which format you're looking at). Just like Format 1, month and day "
    "may each be ONE or TWO digits -- e.g. \"'00430\" (no gaps) is YY=00 "
    "MM=4 DD=30 (April 30, 2000), and \"'00 12 25\" (with gaps) is YY=00 "
    "MM=12 DD=25 (December 25, 2000). The year is always exactly 2 digits "
    "right after the apostrophe; after that, count the remaining digits "
    "carefully to work out where month ends and day begins.\n\n"
    "Whichever format you find, figure out which one it is from where the "
    "apostrophe sits (start of the sequence = Format 2; near the end, "
    "before the last 2 digits = Format 1), then read off the month, day, "
    "and 2-digit year accordingly.\n\n"
    "Find the one corner with the actual stamp and read its digits exactly "
    "as the camera printed them. Respond with ONLY this exact format, "
    "filling in the numbers you read (regardless of which of the two "
    "stamp formats it came from):\n"
    "MM=<number> DD=<number> YY=<number>\n\n"
    "For example: MM=10 DD=19 YY=96\n\n"
    "If none of the four corners has a readable date stamp, respond with "
    "exactly: UNKNOWN\n\n"
    "Do not include any other words, punctuation, letter labels, or "
    "explanation -- just that one line, or UNKNOWN."
)