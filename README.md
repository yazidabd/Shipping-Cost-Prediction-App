# 💰 Shipping Cost Prediction (Machine Learning)

## 🧭 Ringkasan Proyek
Proyek ini bertujuan untuk membangun **model prediksi biaya pengiriman (Shipping Cost Prediction)** berbasis Machine Learning menggunakan dataset hasil pembersihan dari proyek **Logistics Data Analyst** (`logistics_clean.csv`).  
Tujuannya adalah membantu perusahaan logistik memperkirakan **biaya pengiriman** berdasarkan informasi rute, carrier, berat barang, dan jarak pengiriman secara otomatis.  

Hasil akhir model kemudian diintegrasikan ke dalam **aplikasi interaktif berbasis Streamlit** agar pengguna non-teknis dapat melakukan prediksi biaya secara langsung.

---

## 🎯 Tujuan Bisnis
1. Mengembangkan model ML yang mampu memprediksi biaya pengiriman dengan akurasi tinggi.  
2. Mengidentifikasi faktor utama yang paling berpengaruh terhadap biaya.  
3. Menyediakan aplikasi interaktif sederhana untuk simulasi estimasi biaya pengiriman.

---

## 🧮 Tahapan Analisis & Pemodelan (Notebook: `02_modeling_predict_cost.ipynb`)
Berikut adalah ringkasan lengkap proses pemodelan Machine Learning dari awal hingga akhir.

### 1️⃣ Import Library
Menggunakan kombinasi library untuk analisis data, modeling, dan evaluasi performa:
- pandas, numpy, matplotlib, seaborn  
- scikit-learn (Pipeline, ColumnTransformer, train_test_split, metrics)  
- xgboost, RandomForest, LinearRegression  
- RandomizedSearchCV (hyperparameter tuning)  
- joblib (model saving)

---

### 2️⃣ Load Dataset
Dataset `logistics_clean.csv` dimuat sebagai hasil pembersihan dari proyek sebelumnya (Logistics Data Analyst Project).  
Menampilkan `head()` dan distribusi kolom `Status` untuk memastikan integritas data.

---

### 3️⃣ Feature Engineering
Langkah-langkah:
- Pilih fitur utama: `Origin_Warehouse`, `Destination`, `Carrier`, `Shipment_Date`, `Weight_kg`, `Distance_miles`  
- Target: `Cost`  
- Konversi `Shipment_Date` ke datetime  
- Tambah fitur baru `Ship_DayOfWeek`  
- Drop `Shipment_Date` asli  
⚠️ Catatan: untuk menghindari `SettingWithCopyWarning`, gunakan `X = X.copy()` sebelum transformasi.

---

### 4️⃣ Split Data
Pisahkan data menjadi train (80%) dan test (20%) menggunakan `train_test_split` dengan `random_state=42`.

---

### 5️⃣ Preprocessing Pipeline
- Numerical columns: `Weight_kg`, `Distance_miles` → `StandardScaler`  
- Categorical columns: `Origin_Warehouse`, `Destination`, `Carrier`, `Ship_DayOfWeek` → `OneHotEncoder(handle_unknown='ignore')`  
- Digabung menggunakan `ColumnTransformer`.

---

### 6️⃣ Training Baseline Models
Tiga model dicoba:

| Model | RMSE | R² | Catatan |
|-------|------|----|----------|
| Linear Regression | relatif tinggi | moderat | terlalu sederhana |
| Random Forest (200 trees) | lebih rendah | stabil | cukup bagus |
| XGBoost (300 est., lr=0.05) | terbaik | paling tinggi | **dipilih untuk tuning** |

Visualisasi dilakukan dengan scatter actual vs predicted untuk melihat penyebaran hasil prediksi.

---

### 7️⃣ Hyperparameter Tuning
Dilakukan dengan `RandomizedSearchCV` (15 iterasi, 3-fold CV):
- Parameter dicari untuk Random Forest dan XGBoost  
- Kriteria scoring: `r2`  
- Model terbaik: **XGBoost**

**Hasil:**
- `best_params_` dan `best_score_` disimpan.  
- XGBoost dengan kombinasi parameter tertentu menghasilkan peningkatan R² signifikan.

---

### 8️⃣ Evaluasi Model Terbaik
Model terbaik (**XGBoost tuned**):
- RMSE rendah (terbaik di antara semua model)  
- R² tinggi (~0.91)  
- Visualisasi residual menunjukkan error acak (tidak bias)  
- Scatter plot actual vs predicted mendekati garis diagonal.

---

### 9️⃣ Penyimpanan Model
Model disimpan untuk digunakan di aplikasi Streamlit:
```python
import joblib
joblib.dump(best_model, 'model/shipping_cost_predictor.pkl')
```

---

### 🔟 Cara menjalankan app.py
Install requirements.txt
```terminal
pip install -r requirements.txt
```

kemudian, jalankan aplikasi app.py
```terminal
streamlit run app.py
```
