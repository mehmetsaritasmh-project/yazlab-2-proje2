# src/data_loader.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def load_skab_raw(config: dict) -> pd.DataFrame:
    v1_dir = config['paths']['skab_valve1']
    v2_dir = config['paths']['skab_valve2']
    all_dfs = []
    
    for directory, group_name in [(v1_dir, "valve1"), (v2_dir, "valve2")]:
        if not os.path.exists(directory):
            continue
            
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                file_path = os.path.join(directory, file)
                df = pd.read_csv(file_path, sep=';')
                df.columns = [c.lower() for c in df.columns]
                
                # --- ÇÖZÜM: ETİKET SÜTUNUNU BUL ---
                candidates = ['anomaly', 'attack', 'label', 'att_flag']
                found_label = next((c for c in candidates if c in df.columns), None)
                
                if found_label and found_label != 'anomaly':
                    df = df.rename(columns={found_label: 'anomaly'})
                
                # Sütunda -999 varsa ve bunlar aslında 0/1 değilse, veriyi temizle
                # Eğer değerler -999 ise, modelin görmesi için bunları 0 yapıyoruz
                if 'anomaly' in df.columns:
                    df['anomaly'] = df['anomaly'].replace(-999, 0)
                
                df['source_group'] = group_name
                df['source_file'] = file
                all_dfs.append(df)
                
    if not all_dfs:
        raise ValueError("Veri setleri yüklenemedi!")
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # [KONTROL] Etiket dağılımını yazdır
    print(f"\n[DEBUG] Veri seti yüklendi. 'anomaly' sütunu unique değerler: {final_df['anomaly'].unique()}")
    
    return final_df

def get_skab_folds(config: dict) -> list:
    df = load_skab_raw(config)
    
    # 1. 'anomaly' sütununu kesin olarak al
    label_col = 'anomaly'
    
    # Hata kontrolü: Sütun var mı?
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' sütunu bulunamadı! Mevcut: {df.columns.tolist()}")
    
    # Hata kontrolü: Etiketler -999 mu?
    # Eğer öyleyse, muhtemelen 'load_skab_raw' içinde veriyi yanlış işliyorsun.
    if df[label_col].isin([-999]).all():
        raise ValueError("Etiket sütunu sadece -999 içeriyor! Lütfen ham veri yükleme (load_skab_raw) adımını kontrol et.")

    # 2. Feature kolonlarını temizle
    ignored_cols = ['datetime', 'changepoint', 'anomaly', 'source_group', 'source_file']
    feature_cols = [c for c in df.columns if c not in ignored_cols]
    
    X = df[feature_cols].values
    y = df[label_col].values
    groups = df['source_file'].values
    
    # 3. Eğitim
    gkf = GroupKFold(n_splits=5)
    folds = []
    
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        folds.append({
            'X_train': X[train_idx],
            'y_train': y[train_idx],
            'X_test': X[test_idx],
            'y_test': y[test_idx],
            'train_files': groups[train_idx],
            'test_files': groups[test_idx]
        })
    return folds

def load_and_split_batadal(config: dict) -> dict:
    path = config['paths']['batadal_training_2']
    
    # 1. Her durumda tanımlı bir df oluştur
    if not os.path.exists(path):
        print("[Veri Uyarısı] BATADAL dosyası bulunamadı, simüle ediliyor.")
        X_data = np.random.randn(400, 5)
        y_data = np.random.randint(0, 2, 400)
    else:
        # Dosya varsa oku
        try:
            df = pd.read_csv(path, sep=',')
            if len(df.columns) <= 1:
                df = pd.read_csv(path, sep=';')
        except Exception:
            df = pd.read_csv(path)
            
        # [EKLE] Sütunları standartlaştır ve -999 temizle
        df.columns = [c.lower() for c in df.columns]
        
        # Etiket sütununu bul ve -999 temizle
        label_candidates = ['anomaly', 'attack', 'att_flag', 'label']
        label_col = next((c for c in label_candidates if c in df.columns), df.columns[-1])
        df[label_col] = df[label_col].replace(-999, 0)
        
        # Numeric veriyi ayıkla
        numeric_df = df.select_dtypes(include=[np.number])
        if label_col in numeric_df.columns:
            feature_cols = [c for c in numeric_df.columns if c != label_col]
            X_data = numeric_df[feature_cols].values.astype(np.float32)
            y_data = numeric_df[label_col].values.astype(np.int32)
        else:
            X_data = numeric_df.values.astype(np.float32)
            y_data = df[label_col].values.astype(np.int32)

    # Veri bölünmesi (Eğer dosya yoksa X_data zaten yukarıda tanımlandı)
    n = len(X_data)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    return {
        'train': {'X': X_data[:train_end], 'y': y_data[:train_end]},
        'val': {'X': X_data[train_end:val_end], 'y': y_data[train_end:val_end]},
        'test': {'X': X_data[val_end:], 'y': y_data[val_end:]}
    }