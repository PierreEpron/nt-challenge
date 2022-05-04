from typing import Any, Text
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLReader
from pathlib import Path

def convert_path(path : Any) -> Path:
    """
        Shortcut for converting str into path if necessary.
        
        Parameters
        ----------
        path : str or pathlib.Path
            path to convert.
        Return
        ------
        pathlib.Path
            converted path.
    """
    return path if isinstance(path, Path) else Path(path)