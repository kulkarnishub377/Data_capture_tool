import roi

SQUARE = [[100, 100], [300, 100], [300, 300], [100, 300]]


def test_point_inside_and_outside():
    assert roi.point_in_polygon((200, 200), SQUARE) is True
    assert roi.point_in_polygon((50, 50), SQUARE) is False


def test_overlap_ratio_half_outside():
    # box x 200..400 -> half inside the square (100..300)
    r = roi.box_overlap_ratio((200, 150, 400, 250), SQUARE, (640, 480))
    assert 0.45 <= r <= 0.55


def test_overlap_ratio_fully_inside_and_outside():
    assert roi.box_overlap_ratio((150, 150, 250, 250), SQUARE, (640, 480)) > 0.99
    assert roi.box_overlap_ratio((400, 400, 450, 450), SQUARE, (640, 480)) == 0.0


def test_polygon_bbox_clamped_to_frame():
    poly = [[-20, -10], [700, 0], [650, 500]]
    assert roi.polygon_bbox(poly, (640, 480)) == (0, 0, 640, 480)


def test_expand_bbox_adds_margin_and_clamps():
    assert roi.expand_bbox(SQUARE, (640, 480), 0.5) == (0, 0, 400, 400)
    assert roi.expand_bbox(SQUARE, (640, 480), 0.0) == (100, 100, 300, 300)


def test_cached_mask_gives_same_ratio():
    mask = roi.make_mask(SQUARE, (640, 480))
    box = (200, 150, 400, 250)
    assert roi.box_overlap_ratio(box, SQUARE, (640, 480), mask=mask) == \
        roi.box_overlap_ratio(box, SQUARE, (640, 480))


A_POLY = [[10, 10], [50, 10], [50, 50]]
B_POLY = [[200, 200], [300, 200], [300, 300]]


def test_union_bbox_covers_all_polygons():
    assert roi.polygon_bbox([A_POLY, B_POLY], (640, 480)) == (10, 10, 300, 300)


def test_mask_of_two_polygons_covers_both_and_not_between():
    m = roi.make_mask([A_POLY, B_POLY], (640, 480))
    assert m[30, 40] == 1 and m[280, 290] == 1 and m[120, 120] == 0
