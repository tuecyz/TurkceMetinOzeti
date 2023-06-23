from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline
from unidecode import unidecode
import sys
from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import string

from transformers import BartTokenizer, BartForConditionalGeneration
import pdfplumber

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_top_keywords_from_pdf(file_path):
    # PDF'den metin çıkarma
    doc = extract_text_from_pdf(file_path)

    # Türkçe bağlaçları stopwords listesine ekleyin
    stop_words = set(stopwords.words('turkish'))
    stop_words.update(['ve', 'ile', 'veya', 'ise',',','karşı', 'ilk','olan', 'ama', 'fakat', 'lakin', 'çünkü', 'ancak', 'yalnız', 'oysa', 'oysa ki', 'halbuki', 'oysaki' ,'ki', 'de', 'da', 'te', 'ta', 'zira', 'madem', 'mademki', '"','veyahut','%',  'nin','nın','nun','nün','ya da', 'şayet', 'eğer', 'öyleyse', 'öyle', 'son','ön','(',')','arka','sağ','sol','ilklerinden','itibaren','olarak','halinden','halin','sonlarından', 'o halde', 'kısacası', 'demek ki', 'nitekim','acaba','ama', 'ancak','aslında','yalnız','yalnızca','zaten','yani','sonra','tabi','sadece','şimdi','ise', 'neden','oysa','herhangi','hala','az','bazen','belki', 'genellikle','gibi','bile','yoksa', 'biraz','çünkü','daha','değil','anlaşılan','[', ']','/','&','%','+','-','*','^','!','£','#','$','½','ne...ne', 'ya...ya', 'hatta', 'üstelik', 'ayrıca', 'hem', 'hem de', 'yine', 'gene', 'meğer','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'])

    # Kelimeleri tokenize etme ve küçük harfe dönüştürme
    words = [word.lower() for word in word_tokenize(doc)]

    # Porter Stemmer kullanarak stemming yapma
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]

    # WordNet Lemmatizer kullanarak lemmatization yapma
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, wordnet.VERB) for word in words]

    # Cümlelere ayırma
    total_sentences = tokenize.sent_tokenize(doc)
    total_sent_len = len(total_sentences)

    # Türkçe stemmer kullanarak kök bulma
    stemmer = PorterStemmer()

    def preprocess_text(text):
        # Cümlelere ayırma
        sentences = tokenize.sent_tokenize(text)

        # Metin ön işleme adımlarını uygulama
        preprocessed_sentences = []
        for sentence in sentences:
            # Tokenize etme
            words = word_tokenize(sentence)

            # Küçük harflere dönüştürme
            words = [word.lower() for word in words]

            # Noktalama işaretlerini çıkartma
            words = [word for word in words if word not in string.punctuation]

            # Stop kelimeleri çıkartma
            words = [word for word in words if word not in stop_words]

            # Stemming yapma
            words = [stemmer.stem(word) for word in words]

            # Bağlaçları ve ekleri çıkartma
            words = [word for word in words if word not in stop_words]

            # Köklere dönüştürme
            words = [stemmer.stem(word) for word in words]

            preprocessed_sentences.append(words)

        return preprocessed_sentences

    def check_sent(word, sentences):
        final = [all([w in x for w in word]) for x in sentences]
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return len(sent_len)

    # Önişleme adımlarını uygulama
    cleaned_words = []
    for word in words:
        word = word.replace('.', '')
        if word not in stop_words:
            cleaned_words.append(stemmer.stem(word))

    # TF hesaplama
    tf_score = {}
    for word in cleaned_words:
        if word in tf_score:
            tf_score[word] += 1
        else:
            tf_score[word] = 1

    # TF skorlarını normalizasyon
    total_word_length = len(cleaned_words)
    tf_score.update((x, y / total_word_length) for x, y in tf_score.items())

    # IDF hesaplama
    idf_score = {}
    for word in cleaned_words:
        if word in idf_score:
            idf_score[word] = check_sent(word, total_sentences)
        else:
            idf_score[word] = 1

    # IDF skorlarını hesaplama ve güncelleme
    idf_score.update((x, math.log(total_sent_len / (y + 1))) for x, y in idf_score.items())

    # TF-IDF skorlarını hesaplama
    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}

    # En yüksek 5 anahtar kelimeyi alma
    def get_top_n(dict_elem, n):
        result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
        return result

    return get_top_n(tf_idf_score, 5)


