# make copilot importable on Streamlit Cloud
import sys, os
pkg_root = os.path.dirname(__file__)
parent = os.path.dirname(pkg_root)
if parent not in sys.path:
    sys.path.append(parent)

