import sys
try:
    import fer
    print(f"fer file: {fer.__file__}")
    print(f"fer dir: {dir(fer)}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

try:
    from fer import FER
    print("Successfully imported FER class")
except ImportError as e:
    print(f"Failed to import FER: {e}")
