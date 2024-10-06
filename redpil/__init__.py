"""Top-level package for redpil."""

__author__ = 'Mark Harfouche'
__email__ = 'mark.harfouche@gmail.com'

from .redpil import imwrite, imread

__all__ = ['imread', 'imwrite']
from ._version import __version__