"""Static smoke tests: no dataset or TensorFlow execution required."""
from pathlib import Path
import ast

def test_project_python_syntax():
    root = Path(__file__).resolve().parents[1]
    for path in [root/'train.py', *list((root/'fashion_assistant').glob('*.py')), *list((root/'scripts').glob('*.py'))]:
        ast.parse(path.read_text(), filename=str(path))
