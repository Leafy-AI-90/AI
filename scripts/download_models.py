from pathlib import Path

from leafy_ai.vision import download_weights

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    download_weights(root / "models/weights", root / "models/manifest.json")
