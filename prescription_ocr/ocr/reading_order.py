"""
Turning a bag of detected boxes into the order a human would read them.

The detector returns boxes in no useful order. Prescriptions are laid out in
rows (drug, dose, duration side by side), so boxes are grouped into visual rows
first and only then sorted left to right within each row. Sorting purely by y
would interleave columns; sorting purely by x would scramble the page.
"""

# Two boxes belong to the same row if their tops differ by less than this
# fraction of the row's average height.
ROW_GROUPING_FACTOR = 0.5


def sort_reading_order(boxes):
    """
    Sort detection polygons top-to-bottom, then left-to-right within each row.

    `boxes` is a sequence of point lists as returned by the detector; the return
    value is the same boxes, reordered.
    """
    if len(boxes) == 0:
        return []

    # (box, top_y, left_x, height) — everything the grouping needs.
    box_data = []
    for box in boxes:
        y1 = min(p[1] for p in box)
        y2 = max(p[1] for p in box)
        x1 = min(p[0] for p in box)
        box_data.append((box, y1, x1, y2 - y1))

    box_data.sort(key=lambda b: b[1])

    # Walk down the page, starting a new row whenever a box sits too far below
    # the average top of the row being built.
    rows = [[box_data[0]]]
    for item in box_data[1:]:
        avg_y = sum(b[1] for b in rows[-1]) / len(rows[-1])
        avg_h = sum(b[3] for b in rows[-1]) / len(rows[-1])

        if abs(item[1] - avg_y) < avg_h * ROW_GROUPING_FACTOR:
            rows[-1].append(item)
        else:
            rows[-1].sort(key=lambda b: b[2])
            rows.append([item])
    rows[-1].sort(key=lambda b: b[2])

    return [item[0] for row in rows for item in row]
