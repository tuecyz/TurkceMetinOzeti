import re
from nltk import RegexpParser, pos_tag

class AnahtarKelimeler(object):
    def __init__(self, corpus=None, stop_words=[], alpha=0.5):
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("Alpha 0 ile 1 arasında olmalıdır.")
        self.stop_words = stop_words
        stop_word_regex_list = []
        for word in self.stop_words:
            word_regex = r'\b' + word + r'(?![\w-])'
            stop_word_regex_list.append(word_regex)
        self.stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
        self.corpus = corpus
        self.alpha = alpha
        self.parser = RegexpParser('''
                            KEYWORDS: {<DT>? <JJ>* <NN.*>+}
                            P: {<IN>}
                            V: {<V.*>}
                            PHRASES: {<P> <KEYWORDS>}
                            ACTIONS: {<V> <KEYWORDS|PHRASES>*}
                            ''')

    def sayi_mi(self, s):
        try:
            float(s) if '.' in s else int(s)
            return True
        except ValueError:
            return False

    def _cumleleri_ayir(self, metin):
        cumle_bolucu = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013\n]|\\s\\-\\s')
        cumleler = cumle_bolucu.split(metin)
        return cumleler

    def _ifadeleri_ayir(self, cumleler):
        ifade_listesi = []
        for c in cumleler:
            tmp = re.sub(self.stop_word_pattern, '|', c.strip())
            ifadeler = tmp.split("|")
            for ifade in ifadeler:
                ifade = ifade.strip().lower()
                if ifade != "":
                    ifade_listesi.append(ifade)
        ifade_listesi_yeni = []
        for p in ifade_listesi:
            etiketler = pos_tag(self._kelime_ayir(p))
            if etiketler != []:
                parcalar = self.parser.parse(etiketler)
                for subtree in parcalar.subtrees(filter=lambda t: t.label() == 'KEYWORDS'):
                    anahtar_kelime = ' '.join([i[0] for i in subtree])
                    ifade_listesi_yeni.append(anahtar_kelime)

        return ifade_listesi_yeni


    def _kelime_ayir(self, metin):
        bolumleyici = re.compile('[^a-zA-Z0-9_\\+\\-/]')
        kelimeler = []
        for tek_kelime in bolumleyici.split(metin):
            mevcut_kelime = tek_kelime.strip().lower()
            if mevcut_kelime != '' and not self.sayi_mi(mevcut_kelime):
                kelimeler.append(mevcut_kelime)
        return kelimeler

    @property
    def _corpus_anahtar_kelimeleri(self):
        if self.corpus:
            cumleler = self._cumleleri_ayir(self.corpus)
            return self._ifadeleri_ayir(cumleler)
        else:
            return None


    def kelime_skorlarini_hesapla(self, ifade_listesi):
        kelime_sikligi = {}
        kelime_derecesi = {}
        for ifade in ifade_listesi:
            kelime_listesi = self._kelime_ayir(ifade)
            kelime_listesi_uzunlugu = len(kelime_listesi)
            kelime_listesi_derecesi = kelime_listesi_uzunlugu - 1
            for kelime in kelime_listesi:
                kelime_sikligi.setdefault(kelime, 0)
                kelime_sikligi[kelime] += 1
                kelime_derecesi.setdefault(kelime, 0)
                kelime_derecesi[kelime] += kelime_listesi_derecesi
        for madde in kelime_sikligi:
            kelime_derecesi[madde] = kelime_derecesi[madde] + kelime_sikligi[madde]
        kelime_skoru = {}
        for madde in kelime_sikligi:
            kelime_skoru.setdefault(madde, 0)
            kelime_skoru[madde] = kelime_derecesi[madde] / (kelime_sikligi[madde] * 1.0)
        return kelime_skoru


    @property
    def _corpus_kelime_skorlari(self):
        corp_anahtar_kelimeleri = self._corpus_anahtar_kelimeleri
        if corp_anahtar_kelimeleri:
            kelime_skorlari = self.kelime_skorlarini_hesapla(corp_anahtar_kelimeleri)
            anahtar_kelime_adaylari = {}
            for ifade in corp_anahtar_kelimeleri:
                anahtar_kelime_adaylari.setdefault(ifade, 0)
                kelime_listesi = self._kelime_ayir(ifade)
                aday_skoru = 0
                for kelime in kelime_listesi:
                    aday_skoru += kelime_skorlari[kelime]
                anahtar_kelime_adaylari[ifade] = aday_skoru
            return anahtar_kelime_adaylari
        else:
            return None

    def ifade_skorlama(self, ifade_listesi, kelime_skoru):
        corp_skorlari = self._corpus_kelime_skorlari
        anahtar_kelime_adaylari = {}
        for ifade in ifade_listesi:
            anahtar_kelime_adaylari.setdefault(ifade, 0)
            kelime_listesi = self._kelime_ayir(ifade)
            aday_skoru = 0
            for kelime in kelime_listesi:
                aday_skoru += kelime_skoru[kelime]
            if corp_skorlari:
                anahtar_kelime_adaylari[ifade] = (1-self.alpha)*aday_skoru + (self.alpha)*(corp_skorlari[ifade] if ifade in corp_skorlari else 0.0)
            else:
                anahtar_kelime_adaylari[ifade] = aday_skoru
        return anahtar_kelime_adaylari

    def anahtar_kelimeleri_al(self, metin, n=20):
        cumle_listesi = self._cumleleri_ayir(metin)
        ifade_listesi = self._ifadeleri_ayir(cumle_listesi)
        kelime_skorlari = self.kelime_skorlarini_hesapla(ifade_listesi)
        anahtar_kelime_adaylari = self.ifade_skorlama(ifade_listesi, kelime_skorlari)
        return sorted(anahtar_kelime_adaylari.items(), key=lambda x: x[1], reverse=True)[:n]
