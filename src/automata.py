# src/automata.py
import numpy as np
from scipy.stats import norm

class ProbabilisticAutomata:
    def __init__(self, window_size: int, alphabet_size: int):
        """
        Olasılıksal Otomata Yapısı (PAA + SAX + Durum Geçiş Matrisi)
        """
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.cuts = norm.ppf(np.linspace(1/alphabet_size, 1 - 1/alphabet_size, alphabet_size - 1))
        self.alphabet = [chr(97 + i) for i in range(alphabet_size)] # ['a', 'b', 'c', ...]
        
        # Otomata Durum Bilgileri
        self.states_frequency = {}       # Her durumun kaç kez görüldüğü
        self.transition_matrix = {}      # Durum geçiş frekansları: {durum1: {durum2: count}}
        self.transition_probabilities = {} # Olasılık matrisi: {durum1: {durum2: prob}}
        self.all_seen_states = set()

    def _apply_paa(self, data: np.ndarray) -> np.ndarray:
        """
        Zaman serisini belirtilen pencere boyutuna göre ortalama alarak sıkıştırır (PAA).
        """
        # Veri uzunluğu pencere boyutuna tam bölünmeli, kalanı kırpıyoruz
        n = len(data)
        truncated_len = (n // self.window_size) * self.window_size
        reshaped = data[:truncated_len].reshape(-1, self.window_size)
        return np.mean(reshaped, axis=1)

    def _to_sax(self, paa_data: np.ndarray) -> list:
        """
        PAA çıktısı olan sürekli değerleri harflere (sembollere) dönüştürür (SAX).
        """
        sax_sequence = []
        for val in paa_data:
            # Gaussian bölgelerine göre harf ataması
            placed = False
            for idx, cut in enumerate(self.cuts):
                if val < cut:
                    sax_sequence.append(self.alphabet[idx])
                    placed = True
                    break
            if not placed:
                sax_sequence.append(self.alphabet[-1])
        return sax_sequence

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Görülmemiş (unseen) kelimeler için en yakın durumu bulmaya yarayan Edit Distance algoritması.
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def fit(self, data: np.ndarray):
        """
        Eğitim verisi üzerinden PAA ve SAX dönüşümlerini yapıp
        Markov zinciri mantığıyla durum geçiş olasılıklarını hesaplar.
        """
        # 1. Veriyi sembolik hale getir
        paa_res = self._apply_paa(data)
        sax_seq = self._to_sax(paa_res)
        
        # 2. Kaydırılan pencerelerle durum kelimelerini oluştur (Örn: 'aab', 'abc')
        # Kelime uzunluğunu pencere boyutu kadar seçiyoruz
        words = []
        for i in range(len(sax_seq) - self.window_size + 1):
            word = "".join(sax_seq[i:i + self.window_size])
            words.append(word)
            self.all_seen_states.add(word)
            self.states_frequency[word] = self.states_frequency.get(word, 0) + 1

        # 3. Geçiş Frekanslarını Hesapla
        for i in range(len(words) - 1):
            current_state = words[i]
            next_state = words[i+1]
            
            if current_state not in self.transition_matrix:
                self.transition_matrix[current_state] = {}
            
            self.transition_matrix[current_state][next_state] = \
                self.transition_matrix[current_state].get(next_state, 0) + 1

        # 4. Frekansları Olasılığa Dönüştür
        for curr_state, transitions in self.transition_matrix.items():
            total_transitions = sum(transitions.values())
            self.transition_probabilities[curr_state] = {
                nxt_state: count / total_transitions 
                for nxt_state, count in transitions.items()
            }

    def find_nearest_state(self, unseen_state: str) -> str:
        """
        Eğer test verisinde hiç görülmemiş bir durum (kelime) çıkarsa,
        Levenshtein mesafesi en küçük olan en yakın eğitim durumunu döndürür.
        """
        if not self.all_seen_states:
            raise ValueError("Otomata henüz eğitilmemiş!")
            
        best_state = None
        min_distance = float('inf')
        
        for seen_state in self.all_seen_states:
            dist = self._levenshtein_distance(unseen_state, seen_state)
            if dist < min_distance:
                min_distance = dist
                best_state = seen_state
                
        return best_state

    def get_transition_probability(self, current_state: str, next_state: str) -> float:
        """
        İki durum arasındaki geçiş olasılığını döner. Durumlar görülmemişse 
        otomatik olarak Levenshtein eşlemesini devreye sokar.
        """
        # Durum 1 Kontrolü (Unseen ise en yakına eşle)
        resolved_current = current_state
        if current_state not in self.transition_probabilities:
            resolved_current = self.find_nearest_state(current_state)
            
        # Durum 2 Kontrolü (Unseen ise en yakına eşle)
        resolved_next = next_state
        if next_state not in self.all_seen_states:
            resolved_next = self.find_nearest_state(next_state)
            
        # Geçiş kontrolü
        state_transitions = self.transition_probabilities.get(resolved_current, {})
        
        # Eğer bu iki durum arasında eğitimde hiç geçiş olmadıysa çok küçük bir taban olasılık dön (Laplace Smoothing taklidi)
        return state_transitions.get(resolved_next, 1e-5)