def calculate_changed_area(change_mask, pixel_area_m2):
    """
    Calculate physical area represented by changed pixels.

    Parameters
    ----------
    change_mask:
        Boolean change mask.

    pixel_area_m2:
        Ground area represented by one pixel.
    """

    if pixel_area_m2 <= 0:
        raise ValueError("pixel_area_m2 must be greater than zero.")

    changed_pixels = int(change_mask.sum())

    return {
        "changed_pixels": changed_pixels,
        "area_m2": changed_pixels * pixel_area_m2,
    }
