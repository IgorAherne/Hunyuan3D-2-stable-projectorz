# exceptions used by the API, of trellis-stable-projectorz

class CancelledException(Exception):
    """Raised when user requests cancellation (interrupts the generation)."""
    pass