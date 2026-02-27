#!/usr/bin/env python3
"""
Static code structure verification for blop_sim.
Uses AST parsing to verify structure without importing.
"""
import ast
from pathlib import Path

print("=" * 60)
print("Verifying blop_sim code structure")
print("=" * 60)

def check_file_for_class(filepath, class_name):
    """Check if a Python file defines a given class."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read(), filename=str(filepath))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return True
        return False
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return False

def check_file_for_function(filepath, func_name):
    """Check if a Python file defines a given function or method."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read(), filename=str(filepath))
        
        # Check both module-level functions and class methods
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                return True
        return False
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return False

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath) as f:
            ast.parse(f.read(), filename=str(filepath))
        return True
    except SyntaxError as e:
        print(f"  Syntax error in {filepath}: {e}")
        return False

base_path = Path("/workspace/sim/blop_sim")

# Test 1: Check all Python files have valid syntax
print("\n[1] Checking syntax of all Python files...")
python_files = [
    "backends/__init__.py",
    "backends/simple.py",
    "backends/xrt.py",
    "devices/__init__.py",
    "devices/kb_mirror.py",
    "devices/slit.py",
    "devices/detector.py",
    "handlers.py",
    "__init__.py",
]

syntax_ok = True
for pfile in python_files:
    filepath = base_path / pfile
    if not filepath.exists():
        print(f"  ✗ Missing file: {pfile}")
        syntax_ok = False
    elif check_file_syntax(filepath):
        print(f"  ✓ {pfile}")
    else:
        syntax_ok = False

if not syntax_ok:
    print("\n✗ Syntax check failed")
    exit(1)

# Test 2: Check backend classes exist
print("\n[2] Checking backend classes...")
backend_classes = [
    ("backends/__init__.py", "SimBackend"),
    ("backends/simple.py", "SimpleBackend"),
    ("backends/xrt.py", "XRTBackend"),
]

for filepath, classname in backend_classes:
    full_path = base_path / filepath
    if check_file_for_class(full_path, classname):
        print(f"  ✓ Found {classname} in {filepath}")
    else:
        print(f"  ✗ Missing {classname} in {filepath}")
        exit(1)

# Test 3: Check device classes exist
print("\n[3] Checking device classes...")
device_classes = [
    ("devices/kb_mirror.py", "KBMirrorSimple"),
    ("devices/kb_mirror.py", "KBMirrorXRT"),
    ("devices/slit.py", "SlitDevice"),
    ("devices/detector.py", "DetectorDevice"),
    ("devices/detector.py", "SimDetectorController"),
    ("devices/detector.py", "SimDetectorWriter"),
]

for filepath, classname in device_classes:
    full_path = base_path / filepath
    if check_file_for_class(full_path, classname):
        print(f"  ✓ Found {classname} in {filepath}")
    else:
        print(f"  ✗ Missing {classname} in {filepath}")
        exit(1)

# Test 4: Check utility functions exist
print("\n[4] Checking utility functions...")
utils = [
    ("handlers.py", "get_beam_stats"),
]

for filepath, funcname in utils:
    full_path = base_path / filepath
    if check_file_for_function(full_path, funcname):
        print(f"  ✓ Found {funcname} in {filepath}")
    else:
        print(f"  ✗ Missing {funcname} in {filepath}")
        exit(1)

# Test 5: Check key methods in backends
print("\n[5] Checking backend methods...")
backend_methods = [
    ("backends/__init__.py", "register_device"),
    ("backends/__init__.py", "generate_beam"),
    ("backends/__init__.py", "get_image_shape"),
    ("backends/simple.py", "generate_beam"),
    ("backends/xrt.py", "generate_beam"),
]

for filepath, methodname in backend_methods:
    full_path = base_path / filepath
    if check_file_for_function(full_path, methodname):
        print(f"  ✓ Found {methodname} in {filepath}")
    else:
        print(f"  ✗ Missing {methodname} in {filepath}")
        exit(1)

# Test 6: Check detector methods
print("\n[6] Checking detector methods...")
detector_methods = [
    ("devices/detector.py", "observe_indices_written"),
    ("devices/detector.py", "collect_stream_docs"),
    ("devices/detector.py", "_write_single_frame"),
]

for filepath, methodname in detector_methods:
    full_path = base_path / filepath
    if check_file_for_function(full_path, methodname):
        print(f"  ✓ Found {methodname} in {filepath}")
    else:
        print(f"  ✗ Missing {methodname} in {filepath}")
        exit(1)

print("\n" + "=" * 60)
print("All structure checks passed! ✓")
print("=" * 60)
print("\nCode structure is valid. Next steps:")
print("1. Install dependencies with: pixi install")
print("2. Test actual imports and instantiation")
print("3. Run tutorial to verify end-to-end functionality")
