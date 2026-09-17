def format_bytes(num_bytes: int) -> str:
    """Convert a byte count into a human-readable format."""

    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")

    units = ["B", "KB", "MB", "GB", "TB", "PB"]

    size = float(num_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"