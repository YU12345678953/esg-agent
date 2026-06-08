"""
Self-contained ESG report generator package.

The active implementation lives in:
- pipeline.py: PDF parsing, cleaning, chunking, page mapping
- generation.py: ESG section generation
- api.py: FastAPI backend
- web_ui.py: Streamlit frontend
"""

from .generation import SECTION_GROUPS
from .pipeline import run_preprocessing

__all__ = ["SECTION_GROUPS", "run_preprocessing"]
