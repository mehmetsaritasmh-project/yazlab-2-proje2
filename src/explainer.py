# src/explainer.py
import json
import os
import numpy as np
import Levenshtein  # Edit distance için gerekli kütüphane

class AutomataExplainer:
    def __init__(self, automata_model):
        """
        Otomata modellerinin kararlarını matematiksel ve mantıksal olarak
        açıklayan, hocanın rubrik şablonuyla %100 uyumlu akıllı modül.
        """
        self.automata = automata_model

    def explain_sequence(self, raw_subsequence: np.ndarray, scenario_name: str) -> dict:
        """
        Gelen anlık bir zaman serisi diliminin (subsequence) otomata tarafından
        nasıl işlendiğini adım adım analiz eder, edit distance ve güven skorunu hesaplar.
        """
        # 1. Ham veriyi otomataya sokup sembolik harflere dönüştür
        paa_res = self.automata._apply_paa(raw_subsequence)
        sax_seq = self.automata._to_sax(paa_res)
        
        # Pencere boyutuna göre durum kelimelerini oluştur
        words = []
        for i in range(len(sax_seq) - self.automata.window_size + 1):
            word = "".join(sax_seq[i:i + self.automata.window_size])
            words.append(word)
            
        if len(words) < 2:
            return {
                "error": "Zaman serisi dilimi otomata pencere boyutu için çok kısa!"
            }

        steps_log = []
        total_path_probability = 1.0
        unseen_states_encountered = 0
        
        # 2. Durum geçişlerini tek tek incele ve olasılık yollarını hesapla
        for i in range(len(words) - 1):
            curr_state = words[i]
            nxt_state = words[i+1]
            
            # Durumlar eğitimde var mı, yoksa yeni mi (Unseen) tespit et
            is_curr_unseen = curr_state not in self.automata.transition_probabilities
            is_nxt_unseen = nxt_state not in self.automata.all_seen_states
            
            # Levenshtein mesafelerini ve eşlenen durumları hesapla
            resolved_curr = self.automata.find_nearest_state(curr_state) if is_curr_unseen else curr_state
            resolved_nxt = self.automata.find_nearest_state(nxt_state) if is_nxt_unseen else nxt_state
            
            # Edit Distance Hesaplama (Ek Puan Kriteri için)
            curr_distance = Levenshtein.distance(curr_state, resolved_curr) if is_curr_unseen else 0
            nxt_distance = Levenshtein.distance(nxt_state, resolved_nxt) if is_nxt_unseen else 0
            
            if is_curr_unseen or is_nxt_unseen:
                unseen_states_encountered += 1
            
            # Otomatadan geçiş olasılığını al
            transition_prob = self.automata.get_transition_probability(curr_state, nxt_state)
            total_path_probability *= transition_prob
            
            # Hocanın Örnek Açıklama (Bölüm X-E) kriterlerine uygun loglama
            steps_log.append({
                "step": i + 1,
                "transition": f"{curr_state} -> {nxt_state}",
                "status": "unseen" if (is_curr_unseen or is_nxt_unseen) else "seen",
                "current_state_info": {
                    "pattern": curr_state,
                    "status": "unseen" if is_curr_unseen else "seen",
                    "mapped_to": resolved_curr,
                    "edit_distance": curr_distance
                },
                "next_state_info": {
                    "pattern": nxt_state,
                    "status": "unseen" if is_nxt_unseen else "seen",
                    "mapped_to": resolved_nxt,
                    "edit_distance": nxt_distance
                },
                "transition_probability": float(transition_prob)
            })

        # 3. Güven Skoru (Confidence Score) ve Karar Mekanizması
        num_transitions = len(words) - 1
        base_confidence = np.power(total_path_probability, 1 / num_transitions) if num_transitions > 0 else 0
        
        # Unseen durumlar güveni penalize eder
        penalty_factor = 0.7 if unseen_states_encountered > 0 else 1.0
        confidence_score = float(base_confidence * penalty_factor)
        
        # Düşük olasılıklı yollar anomali kabul edilir (Bölüm X-C ve X-E'deki eşik mantığı)
        decision = "anomaly" if confidence_score < 0.2 else "normal"

        # 4. Nihai Rapor Formatı (Bölüm X-F Zorunlu Şablon Alanları Kök Düzeyine Çıkarıldı)
        explanation_report = {
            "scenario": scenario_name,
            "time_step": len(words),  # Simüle edilen adım/uzunluk
            "status": "unseen" if unseen_states_encountered > 0 else "seen",
            "unseen_patterns_detected": unseen_states_encountered,
            "path_probability": float(total_path_probability),
            "confidence_score": round(confidence_score, 4),
            "decision": decision,
            "interpreted_sequence": words,
            "step_by_step_analysis": steps_log
        }
        
        return explanation_report

    def save_explanation_to_json(self, report: dict, filename: str, config: dict):
        """
        Üretilen açıklamayı outputs/explanations/ klasörüne JSON olarak yazar.
        """
        output_dir = os.path.join(config['paths']['output_dir'], "explanations")
        
        # Klasör yoksa otomatik oluştur, hata patlamasın
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
            
        print(f"   [Açıklanabilirlik] Karar raporu başarıyla kaydedildi: {file_path}")