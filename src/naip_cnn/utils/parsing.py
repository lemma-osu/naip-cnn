def float_to_str(f: float) -> str:
    """Stringify a float for a filename.

    For example:
    - 0.5 -> '0p5'
    - 1.0 -> '1'
    """
    f = float(f)
    if f.is_integer():
        return str(int(f))
    return str(f).replace(".", "p")


def str_to_float(s: str) -> float:
    """Parse a string into a float.

    For example:
    - '0p5' -> 0.5
    - '1' -> 1.0
    """
    return float(s.replace("p", "."))
