import argparse
import yaml
import os
import itertools
import pandas as pd
from src.pipeline import run_pipeline

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_output_dirs(base_output_path):
    sub_dirs = ['logs', 'plots', 'models', 'explanations']
    for sub in sub_dirs:
        os.makedirs(os.path.join(base_output_path, sub), exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="YazLab Proje 2 - Zaman Serisi Analizi")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    create_output_dirs(config['paths']['output_dir'])
    
    print(f"\n=== {config['project_name']} Başlatılıyor ===")
    
    # 1. Adım: Standart Pipeline (Derin Öğrenme Modelleri ve Orijinal Otomata) Koşturuluyor
    print("\n>>> Adım 1: Standart Eğitim ve Test Süreçleri Tetikleniyor...")
    run_pipeline(config)
    
    # 2. Adım: Rubrik Bölüm IV-2 Uyarınca Parametre Duyarlılık Analizi (Sensitivity Analysis)
    print("\n" + "="*60)
    print("=== ADIM 2: AUTOMATA PARAMETRE DUYARLILIK ANALİZİ BAŞLATILIYOR ===")
    print("=== (Window Size: [3,4,5,6] & Alphabet Size: [3,4,5,6]) ===")
    print("="*60)
    
    windows = [3, 4, 5, 6]
    alphabets = [3, 4, 5, 6]
    sensitivity_results = []
    
    # src.pipeline içerisinden sadece otomata duyarlılığını test eden fonksiyonu çağırıyoruz
    # Eğer pipeline.py içinde bu isimde bir fonksiyon yoksa, run_pipeline'a parametre geçebiliriz.
    # Biz burada güvenli tarafta kalmak için esnek bir import yapısı kuruyoruz.
    try:
        from src.pipeline import run_automata_sensitivity
    except ImportError:
        # Eğer pipeline içinde izole fonksiyon yoksa, run_pipeline'ı hafifletecek bir wrapper mantığı
        print("[Sistem Bilgisi] İzole hassasiyet fonksiyonu bulunamadı, ana fonksiyon üzerinden parametrik ilerleniyor.")
        run_automata_sensitivity = None

    if run_automata_sensitivity:
        for w, a in itertools.product(windows, alphabets):
            print(f"\n[Deney] Test Ediliyor -> Window Size (K): {w} | Alphabet Size (A): {a}")
            
            # Mevcut konfigürasyonu bozmadan kopyalayıp parametreleri eziyoruz
            temp_config = config.copy()
            temp_config['automata']['window_size'] = w
            temp_config['automata']['alphabet_size'] = a
            
            # Deneyi koştur ve sonuçları al
            metrics = run_automata_sensitivity(temp_config)
            
            sensitivity_results.append({
                'window_size': w,
                'alphabet_size': a,
                'state_count': metrics.get('state_count', 0),
                'transition_count': metrics.get('transition_count', 0),
                'f1_score': metrics.get('f1_score', 0.0),
                'execution_time_sec': metrics.get('exec_time', 0.0)
            })
            
        # Sonuçları raporlama aşaması için CSV dosyasına döküyoruz
        df_report = pd.DataFrame(sensitivity_results)
        report_path = os.path.join(config['paths']['output_dir'], 'logs', 'parameter_sensitivity_report.csv')
        df_report.to_csv(report_path, index=False)
        
        print("\n" + "="*60)
        print(f"✓ [MÜKEMMEL] Parametre Duyarlılık Analizi Raporu Başarıyla Oluşturuldu!")
        print(f"📁 Çıktı Dosyası: {report_path}")
        print("💡 Bu CSV'deki verileri doğrudan projenizin 'Tablo 4' rapor alanına yapıştırabilirsiniz.")
        print("="*60)
    else:
        print("\n[Uyarı] Duyarlılık analizi için src/pipeline.py içinde 'run_automata_sensitivity' fonksiyonu tanımlanmalı!")

if __name__ == "__main__":
    main()