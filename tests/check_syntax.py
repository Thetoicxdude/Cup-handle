
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from pattern_quant.ui.strategy_lab_enhanced import EnhancedStrategyLab
    print("Successfully imported EnhancedStrategyLab")
    
    lab = EnhancedStrategyLab()
    if hasattr(lab, '_render_sweep_results'):
        print("Method _render_sweep_results found")
    else:
        print("ERROR: Method _render_sweep_results NOT found")
        sys.exit(1)
        
    print("Syntax check passed")
except Exception as e:
    print(f"Syntax check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
