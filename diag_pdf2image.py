import os
from pdf2image import convert_from_path

# Try to point pdf2image at the extracted poppler bin
root = os.path.dirname(__file__)
poppler_path = os.path.join(root, 'poppler', 'poppler-23.05.0', 'Library', 'bin')
print('Using poppler_path:', poppler_path, 'exists:', os.path.exists(poppler_path))

try:
    # We won't actually render a PDF; just check that calling the function with poppler_path doesn't error on binaries
    pages = convert_from_path('nonexistent.pdf', first_page=1, last_page=1, poppler_path=poppler_path)
except Exception as e:
    # Expect a file not found error for the PDF, which is fine; if poppler missing, message will indicate
    print('convert_from_path raised:', type(e).__name__, str(e)[:200])
