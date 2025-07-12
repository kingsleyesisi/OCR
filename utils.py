"""
Utility functions and helper classes for the OCR system.

This module contains common utilities, decorators, and helper functions
used across different modules in the application.
"""

import os
import time
import functools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import csv
from datetime import datetime

logger = logging.getLogger(__name__)

class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.operation_name} completed in {duration:.2f} seconds")
    
    @property
    def duration(self) -> float:
        """Get the duration of the timed operation."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    # Remove or replace problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
    safe_name = "".join(c if c in safe_chars else "_" for c in filename)
    
    # Ensure it doesn't start with a dot
    if safe_name.startswith('.'):
        safe_name = 'file_' + safe_name
    
    return safe_name

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def calculate_confidence_level(confidence: float) -> str:
    """
    Determine confidence level based on numeric confidence score.
    
    Args:
        confidence: Numeric confidence score (0-100)
        
    Returns:
        Confidence level string
    """
    if confidence >= 80:
        return "high"
    elif confidence >= 60:
        return "medium"
    else:
        return "low"

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function execution on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

class ResultExporter:
    """Class for exporting OCR results in different formats."""
    
    @staticmethod
    def export_to_txt(text: str, filepath: Path) -> bool:
        """
        Export text to a plain text file.
        
        Args:
            text: Text content to export
            filepath: Output file path
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Text exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export text to {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def export_to_json(data: Dict[str, Any], filepath: Path) -> bool:
        """
        Export data to a JSON file.
        
        Args:
            data: Data dictionary to export
            filepath: Output file path
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data exported to JSON: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export JSON to {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def export_to_csv(data: List[Dict[str, Any]], filepath: Path) -> bool:
        """
        Export data to a CSV file.
        
        Args:
            data: List of data dictionaries to export
            filepath: Output file path
            
        Returns:
            Success status
        """
        try:
            if not data:
                return False
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"Data exported to CSV: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export CSV to {filepath}: {str(e)}")
            return False

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured with level: {log_level}")

def validate_coordinates(coordinates: Tuple[int, int, int, int], 
                        image_width: int, image_height: int) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        coordinates: (x1, y1, x2, y2) coordinates
        image_width: Image width
        image_height: Image height
        
    Returns:
        True if coordinates are valid
    """
    x1, y1, x2, y2 = coordinates
    
    # Check if coordinates are within image bounds
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return False
    
    # Check if coordinates form a valid rectangle
    if x1 >= x2 or y1 >= y2:
        return False
    
    return True

def get_timestamp() -> str:
    """Get current timestamp as formatted string."""
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def bytes_to_mb(bytes_size: int) -> float:
    """Convert bytes to megabytes."""
    return bytes_size / (1024 * 1024)