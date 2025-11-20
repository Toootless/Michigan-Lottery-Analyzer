import shutil, subprocess, sys
print('Python:', sys.executable)
try:
    import pytesseract, pdf2image
    print('pytesseract version:', pytesseract.get_tesseract_version())
except Exception as e:
    print('pytesseract import ok, version check error:', e)
print('pdf2image imported')
# Check if tesseract command is available
exe = shutil.which('tesseract')
print('tesseract exe:', exe)
