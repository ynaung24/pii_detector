"""
Configuration settings for PII Detection Agent.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# PII Detection Configuration
DEFAULT_PROXIMITY_WINDOW = 50  # characters
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
MASK_CHAR = "***"

# File Processing Configuration
MAX_FILE_SIZE_MB = 100
CHUNK_SIZE = 1000  # rows per chunk for large files
