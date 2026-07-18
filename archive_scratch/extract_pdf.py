"""Extract text from PDF for reading."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pdfplumber

pdf_path = r'E:\Master\文献\强化学习\Towardswirelesstime-sensitive networking Multi-link deterministic.pdf'

with pdfplumber.open(pdf_path) as pdf:
    print(f'Total pages: {len(pdf.pages)}')
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            print(f'\n===== PAGE {i+1} =====')
            print(text)
