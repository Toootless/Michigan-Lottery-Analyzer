import os, sys
print('CWD:', os.getcwd())
print('Python executable:', sys.executable)
print('--- Attempting import numpy ---')
try:
    import numpy
    print('numpy version:', numpy.__version__)
    print('numpy.__file__:', numpy.__file__)
except Exception as e:
    print('Numpy import error:', repr(e))
print('\nFirst 20 sys.path entries:')
for p in sys.path[:20]:
    print(' -', p)
print('\nEntries containing "numpy":')
for p in sys.path:
    if p and 'numpy' in p.lower():
        print(' *', p)
