from nltk.corpus import stopwords

from ingilizceOzet import summarize_paper
from ingilizceAnahtar import AnahtarKelimeler


paper_file_path = "C:/Users/user/Desktop/3.SINIF/veriMadenciligi/anahtarKelime/deneme2.pdf"
summarized_paper = summarize_paper(paper_file_path)

anahtar_kelimeleri = AnahtarKelimeler()
keywords = anahtar_kelimeleri.anahtar_kelimeleri_al(paper_file_path, n=10)

stopWords = stopwords.words('english')

# Anahtar kelimeleri hesaplamak
anahtar_kelimeler = AnahtarKelimeler(corpus=summarized_paper, stop_words=stopWords, alpha=0.8)
d = anahtar_kelimeler.anahtar_kelimeleri_al(summarized_paper, n=20)
for i in d:
    print("Anahtar Kelime: %s\nSkor: %f" % (i[0], i[1]))

print("Metin Özeti:")
print(summarized_paper)
