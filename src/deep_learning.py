# src/deep_learning.py
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def set_deterministic_seed(seed: int):
    """
    Deneylerin 5 farklı seed için tamamen tekrarlanabilir ve 
    tutarlı sonuçlar üretmesini garanti eder (Rubrik Bölüm IV-3).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Windows ve GPU ortamları için deterministik operasyonları zorla
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def create_sequence_data(X: np.ndarray, y: np.ndarray, window_size: int):
    """
    Düz zaman serisi verisini Derin Öğrenme modellerinin 
    kabul edeceği (samples, time_steps, features) formatına sokar.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:(i + window_size)])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

def build_model(model_type: str, input_shape: tuple, dropout_rate: float = 0.2):
    """
    Parametrik olarak LSTM veya GRU modeli inşa eder (Rubrik Bölüm II).
    Keras metrik uyuşmazlığını engellemek için sadece temel metrikler derlenir.
    """
    model = Sequential()
    
    if model_type.upper() == "LSTM":
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(32, return_sequences=False))
    elif model_type.upper() == "GRU":
        model.add(GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(GRU(32, return_sequences=False))
    else:
        raise ValueError(f"Desteklenmeyen model tipi: {model_type}. Sadece 'LSTM' veya 'GRU' seçiniz.")
        
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid')) # İkili sınıflandırma (0: Normal, 1: Anomali)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    return model

def train_with_seeds(X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray, config: dict):
    """
    Modelleri eğitir, test seti (val) üzerinde değerlendirir ve 
    hem model yollarını hem de performans metriklerini döner.
    """
    seeds = config['seeds']
    model_types = config['deep_learning']['selected_models']
    epochs = config['deep_learning']['epochs']
    batch_size = config['deep_learning']['batch_size']
    patience = config['deep_learning']['patience']
    output_path = config['paths']['output_dir']
    
    # Pencere boyutunu al
    window_size = config['automata'].get('window_size', 4)
    
    # Zaman serisi verisini hazırla
    X_train_seq, y_train_seq = create_sequence_data(X_train, y_train, window_size)
    X_val_seq, y_val_seq = create_sequence_data(X_val, y_val, window_size)
    
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    all_results = {}

    for model_type in model_types:
        all_results[model_type] = {}
        print(f"\n--- {model_type} Model Eğitimi Başlıyor ---")
        
        for seed in seeds:
            print(f"-> Seed {seed} koşturuluyor...")
            set_deterministic_seed(seed)
            
            model = build_model(model_type, input_shape)
            
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True
            )
            
            model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            # 1. Modeli Kaydet
            model_name = f"{model_type.lower()}_seed{seed}.keras"
            model_save_path = os.path.join(output_path, "models", model_name)
            model.save(model_save_path)
            
            # 2. Modeli Değerlendir (Metrikleri Hesapla)
            # Not: evaluate_model fonksiyonunu burada kullanıyoruz
            metrics = evaluate_model(model, X_val, y_val, window_size)
            
            # 3. Sonuçları Sözlüğe Ekle
            all_results[model_type][seed] = {
                'model_path': model_save_path,
                'metrics': metrics  # Artık metrikler burada!
            }
            print(f"   Seed {seed} eğitimi ve değerlendirmesi tamamlandı.")
            
    return all_results


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, window_size: int) -> dict:
    """
    Hata almamak için pos_label'ı dinamik belirleyen ve 
    çoklu etiketleri güvenli işleyen değerlendirme fonksiyonu.
    """
    X_test_seq, y_test_seq = create_sequence_data(X_test, y_test, window_size)
    
    y_pred_prob = model.predict(X_test_seq, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = y_test_seq.astype(int).flatten()
    
    print(f"DEBUG: y_true - Unique: {np.unique(y_true, return_counts=True)}")
    print(f"DEBUG: y_pred - Unique: {np.unique(y_pred, return_counts=True)}")
    
    # HATA ÇÖZÜMÜ: 
    # Eğer y_true içinde 1 yoksa, pos_label'ı veri setindeki en büyük değer olarak ata
    # veya ikili sınıflandırma için varsayılanı (1) kullanmaya devam et.
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # Eğer 1 etiketi yoksa ve sadece 0 ve -999 varsa, 0'ı pozitif sınıf kabul et
    # Ancak ideal olan, 1'in veri setinde olmasıdır.
    pos_label = 1 if 1 in unique_labels else (0 if 0 in unique_labels else unique_labels[0])

    acc = accuracy_score(y_true, y_pred)
    # average='binary' yerine, çoklu etiket sorunu yaşamamak için 'macro' veya 'weighted' kullanabilirsin
    # ama binary istiyorsan pos_label'ı yukarıdaki gibi dinamik yapmalısın.
    prec = precision_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }