import sys
import platform

print("Python:", sys.version)
print("Platform:", platform.platform())

packages = ['pandas', 'numpy', 'scipy', 'sklearn', 'matplotlib', 'seaborn', 'polars', 'duckdb', 'tqdm', 'torch', 'transformers', 'lightgbm', 'xgboost', 'catboost', 'rapidfuzz', 'Levenshtein']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"  {pkg}: {ver}")
    except ImportError:
        print(f"  {pkg}: NOT installed")
