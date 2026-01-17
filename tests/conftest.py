"""
Pytest configuration for Crisis Detector tests.

This file adds the project root to the Python path so that
the crisis_detector module can be imported during testing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
