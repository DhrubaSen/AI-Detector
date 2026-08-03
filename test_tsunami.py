"""
Quick local test: run analyse_image() on the two tsunami images
and print the results side by side.

Usage:
  1. Place this file in the same folder as main.py (C:\\ai_detector2)
  2. Place the two image files in that same folder
  3. Edit the two filenames below if they don't match exactly
  4. Run:  python test_tsunami.py
"""

from main import analyse_image

FILES = [
    "Tsunami_with_logo.JPG",
    "Tsunami_without_logo.JPG",
]

for filename in FILES:
    print("=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)
    try:
        with open(filename, "rb") as f:
            content = f.read()
        result = analyse_image(content)
        print(f"Label:       {result['label']}")
        print(f"Confidence:  {result['confidence']}%")
        print(f"Explanation: {result['explanation']}")
        print()
        print("AI indicators:")
        for ind in result.get("ai_indicators", []):
            print(f"  - {ind}")
        print("Human indicators:")
        for ind in result.get("human_indicators", []):
            print(f"  - {ind}")
        print()
    except FileNotFoundError:
        print(f"  !! File not found: {filename} — check it's in this folder and the name matches exactly.")
    print()

print("Done.")
