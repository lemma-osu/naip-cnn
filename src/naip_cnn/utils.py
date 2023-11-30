def float_to_str(f: float) -> str:
    """Stringify a float for a filename."""
    f = float(f)
    if f.is_integer():
        return str(int(f))
    return str(f).replace(".", "p")
