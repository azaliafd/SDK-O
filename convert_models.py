"""
Script konversi model .h5 → .keras untuk Skenario 7 (x15 Balanced)
Jalankan SEKALI di terminal sebelum menjalankan app.py:
    python3 convert_models.py
"""

import tensorflow as tf
import os

os.makedirs('models', exist_ok=True)

models_to_convert = [
    ('hasil_training/model_S1_70_15_15.h5', 'models/model_S1.keras'),
    ('hasil_training/model_S2_80_10_10.h5', 'models/model_S2.keras'),
    ('hasil_training/model_S3_70_20_10.h5', 'models/model_S3.keras'),
]

for src, dst in models_to_convert:
    if not os.path.exists(src):
        print(f"⚠️  File tidak ditemukan: {src}")
        continue
    print(f"Memuat: {src} ...")
    model = tf.keras.models.load_model(src)
    print(f"  Input : {model.input_shape} | Output: {model.output_shape}")
    model.save(dst)
    print(f"  ✅ Disimpan: {dst}")

print("\nKonversi selesai. Jalankan: streamlit run app.py")
