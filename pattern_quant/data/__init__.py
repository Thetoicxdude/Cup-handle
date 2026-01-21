"""Data module - 數據攝取層

This module provides data ingestion functionality for the AI PatternQuant system.
"""

from .data_source import DataSource
from .ingestion_service import DataIngestionService

__all__ = [
    'DataSource',
    'DataIngestionService',
]
