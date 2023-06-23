from turkceOzet import showPaperSummary
from turkceAnahtar import extract_top_keywords_from_pdf
import pdfplumber

paperFilePath = "C:/Users/user/Desktop/3.SINIF/veriMadenciligi/anahtarKelime/turkce.pdf"
paperContent = pdfplumber.open(paperFilePath).pages
top_keywords = extract_top_keywords_from_pdf(paperFilePath)
print('Anahtar Kelimeler:')
print(top_keywords)
showPaperSummary(paperContent)
