"""
Turning detected boxes into document reading order, deterministically.

This replaces what the LLM restructurer used to do. Reading order is a geometry
problem: the coordinates are already known, so a layout algorithm answers it
exactly, every time, for any prescription format. An LLM asked the same question
can only guess — and when it guesses wrong it silently reorders or drops
medication lines.

The method is a recursive XY-cut, the standard document-layout decomposition:

  1. Split the page into horizontal bands wherever a full-width strip of
     whitespace runs across it.
  2. Within a band, split into columns wherever a full-height strip of
     whitespace runs down it.
  3. Recurse. A region that can no longer be cut is a leaf.
  4. Within a leaf, group boxes into visual rows by vertical overlap and read
     each row left to right.

Cutting horizontally before vertically is what makes a two-column letterhead
come out as "left column, then right column" instead of zig-zagging between
them — the failure that put a clinic's address in the middle of its doctor's
qualifications.
"""

# Whitespace narrower than this is not a column break. As a fraction of page
# width, so it scales with resolution.
COLUMN_GAP_FRACTION = 0.06

# Whitespace shorter than this is not a band break. As a fraction of the median
# box height, so it scales with handwriting size.
BAND_GAP_FRACTION = 0.4

# Two boxes share a visual row when they overlap vertically by at least this
# fraction of the shorter box's height.
ROW_OVERLAP_FRACTION = 0.3


def _bounds(item):
    """(x1, y1, x2, y2) from a (text, x1, y1, x2, y2) tuple."""
    return item[1], item[2], item[3], item[4]


def _median_height(items):
    heights = sorted(_bounds(i)[3] - _bounds(i)[1] for i in items)
    return heights[len(heights) // 2] if heights else 1


def _merge_spans(spans, min_gap):
    """Merge 1-D spans, treating gaps smaller than min_gap as continuous."""
    spans = sorted(spans)
    merged = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo - merged[-1][1] < min_gap:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def _split(items, axis, min_gap):
    """
    Split items into groups separated by a whitespace gap along `axis`.

    Returns None when the region cannot be cut, which is what ends the
    recursion.
    """
    lo_i, hi_i = (0, 2) if axis == "x" else (1, 3)
    spans = _merge_spans([(_bounds(i)[lo_i], _bounds(i)[hi_i]) for i in items], min_gap)
    if len(spans) <= 1:
        return None

    groups = [[] for _ in spans]
    for item in items:
        b = _bounds(item)
        centre = (b[lo_i] + b[hi_i]) / 2
        # Nearest band by centre, so a box straddling a cut still lands somewhere.
        best = min(range(len(spans)),
                   key=lambda k: 0 if spans[k][0] <= centre <= spans[k][1]
                   else min(abs(centre - spans[k][0]), abs(centre - spans[k][1])))
        groups[best].append(item)
    return [g for g in groups if g]


def group_rows(items, overlap_fraction=ROW_OVERLAP_FRACTION):
    """
    Group boxes into visual rows by vertical overlap, then read left to right.

    Overlap rather than a y-threshold, because doctors write the dose schedule
    slightly below the drug name: those boxes have different tops but clearly
    share a row, and single-linkage on overlap follows that staircase.
    """
    items = sorted(items, key=lambda i: _bounds(i)[1])
    rows, current = [], [items[0]]

    for item in items[1:]:
        top, bottom = _bounds(item)[1], _bounds(item)[3]
        joined = False
        for member in current:
            m_top, m_bottom = _bounds(member)[1], _bounds(member)[3]
            overlap = min(bottom, m_bottom) - max(top, m_top)
            shorter = min(bottom - top, m_bottom - m_top)
            if shorter > 0 and overlap > 0 and overlap / shorter >= overlap_fraction:
                joined = True
                break
        if joined:
            current.append(item)
        else:
            rows.append(sorted(current, key=lambda i: _bounds(i)[0]))
            current = [item]

    rows.append(sorted(current, key=lambda i: _bounds(i)[0]))
    return rows


def _xy_cut(items, column_gap, band_gap, overlap_fraction, depth=0, max_depth=8):
    """Recursive XY-cut. Returns a list of rows, in reading order."""
    if len(items) <= 1:
        return [list(items)]

    if depth < max_depth:
        bands = _split(items, "y", band_gap)
        if bands:
            bands.sort(key=lambda g: min(_bounds(i)[1] for i in g))
            return [row for band in bands
                    for row in _xy_cut(band, column_gap, band_gap,
                                       overlap_fraction, depth + 1, max_depth)]

        columns = _split(items, "x", column_gap)
        if columns:
            columns.sort(key=lambda g: min(_bounds(i)[0] for i in g))
            return [row for column in columns
                    for row in _xy_cut(column, column_gap, band_gap,
                                       overlap_fraction, depth + 1, max_depth)]

    return group_rows(items, overlap_fraction)


def order_rows(items, page_shape,
               column_gap_fraction=COLUMN_GAP_FRACTION,
               band_gap_fraction=BAND_GAP_FRACTION,
               overlap_fraction=ROW_OVERLAP_FRACTION):
    """
    Group (text, x1, y1, x2, y2) items into visual rows, in reading order.

    `page_shape` is the (height, width) of the image the boxes were detected on.
    Returns a list of rows; each row is a list of items ordered left to right.
    """
    if not items:
        return []
    width = page_shape[1]
    return _xy_cut(list(items),
                   column_gap=column_gap_fraction * width,
                   band_gap=band_gap_fraction * _median_height(items),
                   overlap_fraction=overlap_fraction)


def order_items(items, page_shape, **kwargs):
    """The same boxes, flattened into reading order."""
    return [item for row in order_rows(items, page_shape, **kwargs) for item in row]


def merge_rows(items, page_shape, **kwargs):
    """
    Reading-order text lines, one per visual row.

    This is the deterministic stand-in for the LLM restructuring stage: it
    reassembles "Tab. Azithro 500" + "1-0-1" + "x 5d" into one line, in the
    order they appear on the page, without an LLM anywhere near the decision.
    """
    lines = []
    for row in order_rows(items, page_shape, **kwargs):
        text = " ".join(item[0].strip() for item in row if item[0].strip())
        if text:
            lines.append(text)
    return lines
