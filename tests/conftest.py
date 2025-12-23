# pytest configuration for factorium tests
import os
import sys

# Add the project src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
