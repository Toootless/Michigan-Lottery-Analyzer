import re

# Read the file
with open('MichiganLotteryAnalyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the width parameter
content = content.replace("width='stretch'", 'use_container_width=True')

# Write back to file
with open('MichiganLotteryAnalyzer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Fixed button width parameters')