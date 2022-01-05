import os.path
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# Add lib to PYTHONPATH
lib_path = os.path.join(os.path.dirname(__file__), "lib")
add_path(lib_path)
