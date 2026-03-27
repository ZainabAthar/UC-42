import os
import sys
sys.path.append('d:/MD-II/AI-Assisted-Forensic-Analysis-System-for-Decision-Support/BM')
from checks import analyze_document_tampering_fast_ela, call_trufor_analysis

img_path = 'd:/MD-II/AI-Assisted-Forensic-Analysis-System-for-Decision-Support/BM/temp_input/db_qr_page_1.jpg'

res_trufor = call_trufor_analysis(img_path)

with open('test_out.txt', 'w') as f:
    f.write(str(res_trufor))
