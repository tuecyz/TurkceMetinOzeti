from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generate_summary(text, num_sentences):
    # Metinleri vektörize etme
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text])

    # Anahtar kelimeleri belirleme
    feature_names = vectorizer.get_feature_names()
    tfidf_scores = vectors.toarray().flatten()
    top_indices = tfidf_scores.argsort()[:-num_sentences-1:-1]
    keywords = [feature_names[i] for i in top_indices]

    # Metnin cümlelere ayrılması
    sentences = text.split('.')

    # Anahtar kelime içeren cümleleri seçme
    summary_sentences = []
    for sentence in sentences:
        for keyword in keywords:
            if keyword.lower() in sentence.lower():
                summary_sentences.append(sentence.strip())
                break

    # Özeti oluşturma
    summary = ' '.join(summary_sentences)

    return summary

# Örnek metin
text = """
Metin özetleme, bir metnin en önemli bilgilerini daha kısa bir formda sunma işlemidir. Anahtar kelimelerle metnin özetini çıkarmak için TF-IDF ve metin sınıflandırma yöntemleri kullanılabilir. TF-IDF, belirli bir kelimenin belirli bir metindeki önemini ölçen bir istatistiksel ölçüdür. Anahtar kelimeler, metindeki en yüksek TF-IDF skorlarına sahip olan kelimeler olarak belirlenebilir. Ardından, bu anahtar kelimeler kullanılarak metindeki bağlamlarını dikkate alarak özet cümleleri seçebiliriz.
"""

# Metnin özetini oluşturma
summary = generate_summary(text, num_sentences=2)
print("Metin Özeti:")
print(summary)
