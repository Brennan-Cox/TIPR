from contextlib import contextmanager
from io import StringIO
import sys

@contextmanager
def suppress_stdout():
    """
    Method when combined with (with:)
    will not let the code within it's section
    output to standard out
    """
    # Create a StringIO object to capture the output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        # Restore the original standard output
        sys.stdout = old_stdout