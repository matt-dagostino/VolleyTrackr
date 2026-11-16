from utils.bbox_utils import (
    get_centre_of_bbox,
    get_bottom_centre_of_bbox,
    measure_distance,
    bbox_area,
)


def test_bbox_helpers():
    bbox = [10, 20, 30, 40]
    assert get_centre_of_bbox(bbox) == (20.0, 30.0)
    assert get_bottom_centre_of_bbox(bbox) == (20.0, 40.0)
    assert measure_distance((0, 0), (3, 4)) == 5.0
    assert bbox_area(bbox) == 400.0
