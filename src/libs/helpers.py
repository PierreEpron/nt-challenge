from typing import Any, Text
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLReader
from pathlib import Path
from slugify import slugify

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

def load_rasa_data(path : Any, encoding : Text = 'utf-8'):
    """
        Shortcut for load rasa train data with RasaYAMLReader.
        
        Parameters
        ----------
        path : str or pathlib.Path
            path to convert.
        encoding : str, default 'utf-8'
            encoding using when read file
        
        Return
        ------
        pathlib.Path
            converted path.
    """
    # Convert part if necessary
    path = convert_path(path)
    # Read and parse rasa data
    return RasaYAMLReader().reads(path.read_text(encoding=encoding))

def get_slug(text : Text) -> Text:
    """
        Shortcut for slugify text with '_' separator.
        
        Parameters
        ----------
        text : str
            text to slugify.
        
        Return
        ------
        text
            slugified text.
    """
    return slugify(text, separator='_')