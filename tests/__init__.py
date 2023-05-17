# tests/__init__.py
import tempfile
import shutil

def create_temp_dir():
    """Create a temporary directory and return its path."""
    return tempfile.mkdtemp()

def remove_temp_dir(path):
    """Remove a temporary directory and its contents."""
    shutil.rmtree(path)