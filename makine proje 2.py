from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # Sadece Dense kaldı, Input kaldırıldı
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KNeighborsRegressor

# ==============================================================================
# 1. Veri Yükleme ve Temizleme
# ==============================================================================
# Veri setini yükle ve sütun isimlerini temizle (boşluklar, parantezler vb. kaldır)
df = pd.read_csv(Path(__file__).resolve().parent / 'CO2 Emissions_Canada.csv')
df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '_').str.replace('.', '').str.strip()

# Eksik değerleri kontrol et
print("Eksik değerler:")
print(df.isnull().sum())

# Sayısal ve kategorik sütunları ayır, hedef değişkeni belirle
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()
target_col = 'CO2_Emissionsg_km'
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

print(f"\nModel için Sayısal Özellikler: {numeric_cols}")
print(f"Model için Kategorik Özellikler: {categorical_cols}")
print(f"Hedef Değişken: {target_col}")

# Özellik (X) ve hedef (y) değişkenlerini ayır
X = df.drop(target_col, axis=1)
y = df[target_col]

# ==============================================================================
# 2. Veri Ön İşleme
# ==============================================================================
# Kategorik değişkenler için One-Hot Encoding uygula
print("\nKategorik değişkenler için One-Hot Encoding uygulanıyor...")
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print("One-Hot Encoding Tamamlandı.")
print(f"One-Hot Encoding sonrası özellik sayısı: {X.shape[1]}")

# Sayısal özellikleri ölçeklendir (StandardScaler ile)
print("Sayısal Özellikler Ölçeklendiriliyor...")
scaler = StandardScaler()
cols_to_scale_in_X = [col for col in numeric_cols if col in X.columns]
X[cols_to_scale_in_X] = scaler.fit_transform(X[cols_to_scale_in_X])
joblib.dump(scaler,"scaler.pkl")
print("Ölçeklendirme Tamamlandı.")

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nEğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")
print("Veri ön işleme adımları tamamlandı. Şimdi modelleme aşamasına geçebiliriz.")

# ==============================================================================
# 3. Model Tanımlama ve Optimizasyon
# ==============================================================================

# --- 3.1. LightGBM Modeli ---
print("\nLightGBM Hiperparametre Optimizasyonu Başlatılıyor (RandomizedSearchCV)...")
param_distributions_lgbm = {
    'num_leaves': randint(20, 100),
    'learning_rate': uniform(loc=0.01, scale=0.15),
    'n_estimators': randint(100, 500),
    'max_depth': randint(-1, 20),
    'reg_alpha': uniform(loc=0, scale=1),
    'reg_lambda': uniform(loc=0, scale=1),
    'colsample_bytree': uniform(loc=0.6, scale=0.4),
    'subsample': uniform(loc=0.6, scale=0.4)
}
lgb_estimator = lgb.LGBMRegressor(objective='regression', metric='mse', verbose=-1, random_state=42)
random_search_lgbm = RandomizedSearchCV(
    estimator=lgb_estimator,
    param_distributions=param_distributions_lgbm,
    n_iter=50,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search_lgbm.fit(X_train, y_train)
print(f"\nEn iyi R2 Skoru (LightGBM - RandomizedSearchCV): {random_search_lgbm.best_score_:.4f}")
print("En iyi LightGBM parametreleri:", random_search_lgbm.best_params_)
lgbm_model = random_search_lgbm.best_estimator_

# LightGBM performansını değerlendir (YSA ve KNN gibi ayrı başlık altında)
y_train_pred_lgbm = lgbm_model.predict(X_train)
y_test_pred_lgbm = lgbm_model.predict(X_test)
train_r2_lgbm, train_mse_lgbm, train_mae_lgbm = [r2_score(y_train, y_train_pred_lgbm),
                                                mean_squared_error(y_train, y_train_pred_lgbm),
                                                mean_absolute_error(y_train, y_train_pred_lgbm)]
test_r2_lgbm, test_mse_lgbm, test_mae_lgbm = [r2_score(y_test, y_test_pred_lgbm),
                                              mean_squared_error(y_test, y_test_pred_lgbm),
                                              mean_absolute_error(y_test, y_test_pred_lgbm)]

print(f"\nLightGBM Eğitim Seti Performansı:")
print(f"  R2 Score: {train_r2_lgbm:.4f}")
print(f"  MSE: {train_mse_lgbm:.4f}")
print(f"  MAE: {train_mae_lgbm:.4f}")

print(f"\nLightGBM Test Seti Performansı:")
print(f"  R2 Score: {test_r2_lgbm:.4f}")
print(f"  MSE: {test_mse_lgbm:.4f}")
print(f"  MAE: {test_mae_lgbm:.4f}")


# --- 3.2. Yapay Sinir Ağı (YSA) Modeli ---
print("\n========== Yapay Sinir Ağı (YSA) Modeli ==========")
model_nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)), # input_shape ilk Dense katmanına eklendi
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Regresyon için tek çıkış
])
model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stopping_nn = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
print("Yapay Sinir Ağı eğitimi başlatılıyor...")
history_nn = model_nn.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping_nn],
    verbose=0
)
print("Yapay Sinir Ağı eğitimi tamamlandı.")

# YSA performansını değerlendir
y_train_pred_nn = model_nn.predict(X_train).flatten()
y_test_pred_nn = model_nn.predict(X_test).flatten()
train_r2_nn, train_mse_nn, train_mae_nn = [r2_score(y_train, y_train_pred_nn),
                                           mean_squared_error(y_train, y_train_pred_nn),
                                           mean_absolute_error(y_train, y_train_pred_nn)]
test_r2_nn, test_mse_nn, test_mae_nn = [r2_score(y_test, y_test_pred_nn),
                                        mean_squared_error(y_test, y_test_pred_nn),
                                        mean_absolute_error(y_test, y_test_pred_nn)]

print(f"\nYapay Sinir Ağı Eğitim Seti Performansı:")
print(f"  R2 Score: {train_r2_nn:.4f}")
print(f"  MSE: {train_mse_nn:.4f}")
print(f"  MAE: {train_mae_nn:.4f}")

print(f"\nYapay Sinir Ağı Test Seti Performansı:")
print(f"  R2 Score: {test_r2_nn:.4f}")
print(f"  MSE: {test_mse_nn:.4f}")
print(f"  MAE: {test_mae_nn:.4f}")

# --- 3.3. K-En Yakın Komşu (KNN) Modeli ---
print("\n========== K-En Yakın Komşu (KNN) Modeli ==========")
param_distributions_knn = {
    'n_neighbors': randint(3, 20),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
knn_estimator = KNeighborsRegressor(n_jobs=-1)
random_search_knn = RandomizedSearchCV(
    estimator=knn_estimator,
    param_distributions=param_distributions_knn,
    n_iter=30,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
print("KNN Hiperparametre Optimizasyonu Başlatılıyor (RandomizedSearchCV)...")
random_search_knn.fit(X_train, y_train)
print(f"\nEn iyi R2 Skoru (KNN - RandomizedSearchCV): {random_search_knn.best_score_:.4f}")
print("En iyi KNN parametreleri:", random_search_knn.best_params_)
knn_model = random_search_knn.best_estimator_

# KNN performansını değerlendir
y_train_pred_knn = knn_model.predict(X_train)
y_test_pred_knn = knn_model.predict(X_test)
train_r2_knn, train_mse_knn, train_mae_knn = [r2_score(y_train, y_train_pred_knn),
                                              mean_squared_error(y_train, y_train_pred_knn),
                                              mean_absolute_error(y_train, y_train_pred_knn)]
test_r2_knn, test_mse_knn, test_mae_knn = [r2_score(y_test, y_test_pred_knn),
                                           mean_squared_error(y_test, y_test_pred_knn),
                                           mean_absolute_error(y_test, y_test_pred_knn)]

print(f"\nKNN Eğitim Seti Performansı:")
print(f"  R2 Score: {train_r2_knn:.4f}")
print(f"  MSE: {train_mse_knn:.4f}")
print(f"  MAE: {train_mae_knn:.4f}")

print(f"\nKNN Test Seti Performansı:")
print(f"  R2 Score: {test_r2_knn:.4f}")
print(f"  MSE: {test_mse_knn:.4f}")
print(f"  MAE: {test_mae_knn:.4f}")

# ==============================================================================
# 4. Overfitting Analizi ve Görselleştirme
# ==============================================================================
print("\n========== Aşırı Öğrenme (Overfitting) Analizi - Tüm Modeller ==========")

# LightGBM
# Bu metrikler artık yukarıdaki LightGBM bölümünde hesaplandığı için burada tekrar hesaplamaya gerek yok
# Ancak yorumlar için değerler hala kullanılıyor.
print("\n--- LightGBM Aşırı Öğrenme ---")
print(f"  Eğitim R2: {train_r2_lgbm:.4f}, Test R2: {test_r2_lgbm:.4f} (Fark: {train_r2_lgbm - test_r2_lgbm:.4f})")
print(f"  Eğitim MSE: {train_mse_lgbm:.4f}, Test MSE: {test_mse_lgbm:.4f} (Fark: {train_mse_lgbm - test_mse_lgbm:.4f})")
if (train_r2_lgbm - test_r2_lgbm) > 0.01 or (train_mse_lgbm - test_mse_lgbm) > 10:
    print("  Yorum: LightGBM'de potansiyel aşırı öğrenme işaretleri var.")
else:
    print("  Yorum: LightGBM, tutarlı performans gösteriyor, aşırı öğrenme eğilimi düşük.")

# YSA
print("\n--- Yapay Sinir Ağı Aşırı Öğrenme ---")
print(f"  Eğitim R2: {train_r2_nn:.4f}, Test R2: {test_r2_nn:.4f} (Fark: {train_r2_nn - test_r2_nn:.4f})")
print(f"  Eğitim MSE: {train_mse_nn:.4f}, Test MSE: {test_mse_nn:.4f} (Fark: {train_mse_nn - test_mse_nn:.4f})")
if (train_r2_nn - test_r2_nn) > 0.01 or (train_mse_nn - test_mse_nn) > 10:
    print("  Yorum: YSA'da potansiyel aşırı öğrenme işaretleri var.")
else:
    print("  Yorum: YSA, tutarlı performans gösteriyor, aşırı öğrenme eğilimi düşük.")

# KNN
print("\n--- KNN Aşırı Öğrenme ---")
print(f"  Eğitim R2: {train_r2_knn:.4f}, Test R2: {test_r2_knn:.4f} (Fark: {train_r2_knn - test_r2_knn:.4f})")
print(f"  Eğitim MSE: {train_mse_knn:.4f}, Test MSE: {test_mse_knn:.4f} (Fark: {train_mse_knn - test_mse_knn:.4f})")
if (train_r2_knn - test_r2_knn) > 0.01 or (train_mse_knn - test_mse_knn) > 10:
    print("  Yorum: KNN'de potansiyel aşırı öğrenme işaretleri var, 'distance' veya küçük 'n_neighbors' olabilir.")
else:
    print("  Yorum: KNN, tutarlı performans gösteriyor, aşırı öğrenme eğilimi düşük.")
print("=========================================================")

# Görsel Overfitting Analizi (Gerçek vs. Tahmin)
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
fig.suptitle('Modellerin Gerçek vs. Tahmin Değerleri (Test Seti)', fontsize=16)
for ax, model, train_pred, test_pred, title in zip(
    axes, [lgbm_model, model_nn, knn_model],
    [y_train_pred_lgbm, y_train_pred_nn, y_train_pred_knn],
    [y_test_pred_lgbm, y_test_pred_nn, y_test_pred_knn],
    ['LightGBM', 'Yapay Sinir Ağı', 'KNN']
):
    ax.scatter(y_train, train_pred, alpha=0.5, label='Eğitim Tahminleri', color='blue')
    ax.scatter(y_test, test_pred, alpha=0.5, label='Test Tahminleri', color='red')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_title(title)
    ax.set_xlabel(f'Gerçek {target_col}')
    ax.set_ylabel(f'Tahmin {target_col}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Görsel Overfitting Analizi (Artık Dağılımı)
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
fig.suptitle('Modellerin Artık (Residual) Dağılımları (Test Seti)', fontsize=16)
for ax, residuals, title in zip(
    axes, [y_test - y_test_pred_lgbm, y_test - y_test_pred_nn, y_test - y_test_pred_knn],
    ['LightGBM', 'Yapay Sinir Ağı', 'KNN']
):
    sns.histplot(residuals, bins=30, kde=True, ax=ax)
    ax.set_title(f'{title} Artık Dağılımı')
    ax.set_xlabel('Hata (Gerçek - Tahmin)')
    ax.set_ylabel('Frekans')
    ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ==============================================================================
# 5. Model Karşılaştırma ve Kaydetme
# ==============================================================================
print("\n========== Model Karşılaştırma Özeti ==========")
comparison_df = pd.DataFrame({
    'Model': ['LightGBM', 'Yapay Sinir Ağı', 'KNN'],
    'Eğitim R2': [train_r2_lgbm, train_r2_nn, train_r2_knn],
    'Test R2': [test_r2_lgbm, test_r2_nn, test_r2_knn],
    'Eğitim MSE': [train_mse_lgbm, train_mse_nn, train_mse_knn],
    'Test MSE': [test_mse_lgbm, test_mse_nn, test_mse_knn],
    'Eğitim MAE': [train_mae_lgbm, train_mae_nn, train_mae_knn], # Buradaki MAE değerleri güncellendi
    'Test MAE': [test_mae_lgbm, test_mae_nn, test_mae_knn]       # Buradaki MAE değerleri güncellendi
})
print(comparison_df.round(4).to_string(index=False))

print("\nModel Yorumu:")
print("- **LightGBM:** Hızlı ve yüksek performanslı, hiperparametre optimizasyonu ile başarılı, overfitting düşük.")
print("- **Yapay Sinir Ağı:** Karmaşık ilişkileri öğrenir, ölçeklendirmeye duyarlı, erken durdurma ile kontrol.")
print("- **KNN:** Basit ve sezgisel, küçük veri için iyi, ama yüksek boyutlu veride overfitting riski.")

# Modelleri kaydet
joblib.dump(lgbm_model, "lgbm_optimized_model.pkl")
model_nn.save("nn_model.keras")  # Native Keras formatı
joblib.dump(knn_model, "knn_optimized_model.pkl")
joblib.dump(X.columns, "model_features.pkl")
print("\nModeller ve özellikler başarıyla kaydedildi.")

# ==============================================================================
# 6. Ek Görselleştirmeler
# ==============================================================================
# Korelasyon Matrisi
df_for_corr = X.copy()
df_for_corr[target_col] = y
plt.figure(figsize=(15, 12))
corr_matrix_processed = df_for_corr.select_dtypes(include=np.number).corr()
sns.heatmap(corr_matrix_processed, annot=True, fmt=".2f", cmap='coolwarm', vmax=1, vmin=-1, center=0)
plt.title('Ön İşlenmiş Verideki Sayısal Özellikler Arası Korelasyon Matrisi (Hedef Dahil)')
plt.show()

# Özellik Önemliliği (LightGBM)
plt.figure(figsize=(10, 6))
lgb.plot_importance(lgbm_model, max_num_features=10, importance_type='split')
plt.title('LightGBM Özellik Önemliliği (Top 10)')
plt.show()