import re
import requests
import wikipedia
from bs4 import BeautifulSoup
import html2text

try:
	from marker.converters.pdf import PdfConverter
	from marker.models import create_model_dict
	from marker.output import text_from_rendered
except ImportError:
	PdfConverter = None
	create_model_dict = None
	text_from_rendered = None

from markitdown import MarkItDown
from simhash import Simhash
from sentence_transformers import SentenceTransformer
from readability import Document