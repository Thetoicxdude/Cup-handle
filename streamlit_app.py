"""Streamlit Cloud Entry Point for AI PatternQuant

This is the main entry point for Streamlit Cloud deployment.
Streamlit Cloud automatically looks for streamlit_app.py in the repository root.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Debug: Check if modules are available
try:
    import plotly
    print(f"✓ plotly {plotly.__version__} is installed")
except ImportError as e:
    print(f"✗ plotly import failed: {e}")

try:
    from pattern_quant.evolution import EvolutionaryEngine
    print(f"✓ evolution module is available")
except ImportError as e:
    print(f"✗ evolution module import failed: {e}")

try:
    from pattern_quant.strategy.dual_engine import DualEngineStrategy
    print(f"✓ dual_engine module is available")
except ImportError as e:
    print(f"✗ dual_engine module import failed: {e}")

# Import and run the main application
from pattern_quant.ui.app import main

if __name__ == "__main__":
    main()
