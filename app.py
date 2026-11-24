import streamlit as st
import pandas as pd
import joblib
from preprocessing import build_model_input
from tensorflow.keras.models import load_model # Keras modelini yüklemek için

# 1. Kaydedilmiş Modelleri ve Preprocessor'ları Yükle
# @st.cache_resource, modelleri ve scaler'ı sadece bir kez yüklemeyi sağlar, uygulama yenilendiğinde tekrar yüklenmez.
@st.cache_resource
def load_all_assets():
    lgbm_model = joblib.load("lgbm_optimized_model.pkl")
    nn_model = load_model("nn_model.keras")
    knn_model = joblib.load("knn_optimized_model.pkl")
    scaler = joblib.load("scaler.pkl") # ÖNEMLİ: Ana kodunuzda scaler'ı kaydettiğinizden emin olun!
    model_features = joblib.load("model_features.pkl") # Model eğitilirken kullanılan tüm özelliklerin sırasını tutar

    # Orijinal veri setinden benzersiz kategorik değerleri çekmek için
    # Bu kısım, input alanlarındaki seçenekleri doldurmak için kritik
    # CO2 Emissions_Canada.csv dosyanızın doğru yolda olduğundan emin olun
    try:
        df_original = pd.read_csv("CO2 Emissions_Canada.csv")
        df_original.columns = df_original.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '_').str.replace('.', '').str.strip()

        # Hedef değişkeni belirleyin (CO2_Emissionsg_km)
        target_col = 'CO2_Emissionsg_km'
        if target_col in df_original.columns:
            X_original_for_dummies = df_original.drop(target_col, axis=1)
        else:
            X_original_for_dummies = df_original.copy() # Eğer hedef değişken yoksa tüm df'yi kullan

        # Kategorik sütunları tanımlayın
        categorical_cols = ['Make', 'Model', 'Vehicle_Class', 'Transmission', 'Fuel_Type'] # Bu isimler veri setinizdeki orijinal sütun adları olmalı

        # Benzersiz değerleri çekin
        unique_makes = sorted(df_original['Make'].unique().tolist())
        unique_models = sorted(df_original['Model'].unique().tolist())
        unique_vehicle_classes = sorted(df_original['Vehicle_Class'].unique().tolist())
        unique_transmissions = sorted(df_original['Transmission'].unique().tolist())
        unique_fuel_types = sorted(df_original['Fuel_Type'].unique().tolist())

    except FileNotFoundError:
        st.error("Veri seti 'CO2 Emissions_Canada.csv' bulunamadı. Lütfen aynı dizinde olduğundan emin olun.")
        st.stop() # Uygulamayı durdur

    return lgbm_model, nn_model, knn_model, scaler, model_features, \
           unique_makes, unique_models, unique_vehicle_classes, \
           unique_transmissions, unique_fuel_types, categorical_cols # categorical_cols da döndürülüyor

# Tüm varlıkları yükle
lgbm_model, nn_model, knn_model, scaler, model_features_list, \
unique_makes, unique_models, unique_vehicle_classes, \
unique_transmissions, unique_fuel_types, all_categorical_cols = load_all_assets()

# Numeric sütun adlarını global olarak tanımla (ana eğitim kodunuzdakiyle aynı olmalı)
numeric_cols = ['Engine_SizeL', 'Cylinders', 'Fuel_Consumption_City_L_100_km',
                'Fuel_Consumption_Hwy_L_100_km', 'Fuel_Consumption_Comb_L_100_km',
                'Fuel_Consumption_Comb_mpg']


# 2. Arayüz Başlığı
st.title("CO2 Emisyon Tahmin Uygulaması")
st.write("Araç özelliklerini girerek CO2 emisyonunu tahmin edin.")

# 3. Model Seçimi (Sol kenar çubuğunda)
st.sidebar.header("Model Seçimi")
selected_model_name = st.sidebar.radio("Tahmin için modeli seçin:", ('LightGBM', 'Yapay Sinir Ağı (NN)', 'KNN'))

selected_model = None
if selected_model_name == 'LightGBM':
    selected_model = lgbm_model
elif selected_model_name == 'Yapay Sinir Ağı (NN)':
    selected_model = nn_model
else: # KNN
    selected_model = knn_model

# 4. Giriş Alanları (Ana bölümde)
st.header("Araç Özellikleri")

# Sayısal girişler
col1, col2 = st.columns(2)
with col1:
    engine_size = st.number_input("Motor Hacmi (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    cylinders = st.number_input("Silindir Sayısı", min_value=2, max_value=16, value=4, step=1)
    fuel_city = st.number_input("Yakıt Tüketimi (Şehir - L/100km)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
with col2:
    fuel_hwy = st.number_input("Yakıt Tüketimi (Otoban - L/100km)", min_value=0.0, max_value=25.0, value=8.0, step=0.1)
    fuel_comb_L = st.number_input("Yakıt Tüketimi (Kombine - L/100km)", min_value=0.0, max_value=28.0, value=9.0, step=0.1)
    fuel_comb_mpg = st.number_input("Yakıt Tüketimi (Kombine - mpg)", min_value=0.0, max_value=60.0, value=30.0, step=0.1)

col3, col4 = st.columns(2)
with col3:
    make = st.selectbox("Marka", unique_makes)
    vehicle_class = st.selectbox("Araç Sınıfı", unique_vehicle_classes)
    transmission = st.selectbox("Vites Tipi", unique_transmissions)
with col4:
    model_val = st.selectbox("Model", unique_models)
    fuel_type = st.selectbox("Yakıt Tipi", unique_fuel_types)


# 5. Tahmin Butonu
if st.button("CO2 Emisyonunu Tahmin Et"):
    # Girişleri DataFrame'e dönüştür
    # Önce tüm sayısal ve kategorik sütunları içeren boş bir DataFrame oluşturun
    # Bu, One-Hot Encoding sonrası tüm beklenen sütunların olmasını sağlar
    input_df = pd.DataFrame(columns=[
        'Engine_SizeL', 'Cylinders', 'Fuel_Consumption_City_L_100_km',
        'Fuel_Consumption_Hwy_L_100_km', 'Fuel_Consumption_Comb_L_100_km',
        'Fuel_Consumption_Comb_mpg', 'Make', 'Model', 'Vehicle_Class',
        'Transmission', 'Fuel_Type'
    ])

    # Yeni giriş değerlerini DataFrame'e ekle
    input_df.loc[0] = [
        engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb_L, fuel_comb_mpg,
        make, model_val, vehicle_class, transmission, fuel_type
    ]

    # Kategorik değişkenler için One-Hot Encoding uygula
    # `all_categorical_cols` listesi, `load_all_assets` fonksiyonundan geliyor ve doğru kategorik sütun isimlerini içeriyor.
    final_input = build_model_input(input_df, all_categorical_cols, model_features_list)

    # Sayısal sütunları ölçeklendir
    # numeric_cols listesi global olarak tanımlanmış olmalı
    cols_to_scale_in_input = [col for col in numeric_cols if col in final_input.columns]
    final_input[cols_to_scale_in_input] = scaler.transform(final_input[cols_to_scale_in_input])

    # Tahmin yap
    if selected_model_name == 'Yapay Sinir Ağı (NN)':
        prediction = selected_model.predict(final_input)[0][0] # YSA'nın tahmin çıktısı 2D dizi olabilir
    else:
        prediction = selected_model.predict(final_input)[0]

    st.success(f"Tahmin Edilen CO2 Emisyonu: {prediction:.2f} g/km")

st.markdown("---")
st.markdown("Proje: CO2 Emisyon Tahmini | Modeller: LightGBM, Yapay Sinir Ağı, KNN")
st.markdown("Anıl Polat Tarafından Oluşturuldu.")