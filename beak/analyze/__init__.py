""" Analyzes molecular dynamics simulations """

__version__ = '0.0.0a1'
__author__  = 'Robin Betz'

from .Analyzer import Analyzer
from .inDistanceAnalyzer import MinDistanceAnalyzer
from .RmsdAnalyzer import RmsdAnalyzer
from .DihedralAnalyzer import DihedralAnalyzer