import streamlit as st
import joblib
import pandas as pd

# =========================
# Setup Halaman
# =========================
st.set_page_config(page_title="Shipping Cost Dashboard", page_icon="🚚", layout="centered")
st.markdown("<h2 style='text-align:center;color:#D3DAD9;'>🚚 XGBoost Shipping Cost Prediction</h2>", unsafe_allow_html=True)
st.divider()

st.markdown(
    """
    <style>
    /* ===== Ubah warna background utama ===== */
    .stApp {
        background-color: #37353E; /* biru muda lembut */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# =========================
# Load Model & Data Lookup
# =========================
model = joblib.load("../model/shipping_cost_xgb_tuned.pkl")
lookup = pd.read_csv("../dataset/warehouse_distance_lookup.csv")

# Pastikan kolom sesuai
lookup = lookup[['Origin_Warehouse', 'Destination', 'Distance_miles']].drop_duplicates()

# =========================
# Input Pengguna
# =========================
col1, col2 = st.columns(2)
with col1:
    origin = st.selectbox(
        "Warehouse", 
        sorted(lookup['Origin_Warehouse'].unique())
    )

    ship_day = st.selectbox(
        "delivery day", 
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    weight = st.number_input("Weight (kg)", max_value=1000.0, step=0.1, format="%.1f")

with col2:
    # filter tujuan yang valid sesuai origin terpilih
    possible_dest = lookup[lookup['Origin_Warehouse'] == origin]['Destination'].unique()
    destination = st.selectbox("Destination", sorted(possible_dest))

    carrier = st.selectbox(
        "Carrier", 
        ["UPS", "FedEx", "DHL", "OnTrac", "Amazon Logistics", "LaserShip", "USPS"]
    )

    # ambil semua jarak yang mungkin untuk origin-destination ini
    possible_distances = lookup[
        (lookup['Origin_Warehouse'] == origin) & 
        (lookup['Destination'] == destination)
    ]['Distance_miles'].unique()

    distance = st.selectbox("Distance (miles)", sorted(possible_distances))

# =========================
# Buat DataFrame Input
# =========================
input_data = pd.DataFrame({
    "Weight_kg": [weight],
    "Distance_miles": [distance],
    "Carrier": [carrier],
    "Origin_Warehouse": [origin],
    "Destination": [destination],
    "Ship_DayOfWeek": [ship_day]
})

# =========================
# Tombol Prediksi (Tengah)
# =========================
# Jarak sedikit di atas
st.markdown("<br>", unsafe_allow_html=True)

# Layout 3 kolom
col1, col2, col3 = st.columns([1, 2, 1])  # kolom tengah lebih lebar
with col2:
    # Tombol penuh & tidak terpotong
    predict_button = st.button("Cost prediction", use_container_width=True)

# =========================
# Hasil Prediksi
# =========================
if predict_button:
    # 🔍 Validasi: berat tidak boleh 0
    if weight < 1:
        st.warning("⚠️ The weight of the item cannot be 0 kg. Please fill in the weight first.")
    else:
        pred = model.predict(input_data)[0]

        st.markdown("<hr style='border:1px solid #44444E;'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style='text-align:center;'>
                <p style='font-size:22px; color:#D3DAD9; font-weight:bold;'>
                    Estimated Shipping Costs
                </p>
                <p style='font-size:36px; color:#FCB53B; font-weight:bold; margin-top:-10px;'>
                    ${pred:,.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <p style='text-align:center; color:gray; font-size:14px; margin-top:-5px;'>
            Model: <b>XGBoost Tuned</b> | R² = 0.939 | RMSE = 23.37
            </p>
            """, 
            unsafe_allow_html=True
        )
        with st.expander("📋 Detail Input yang Digunakan"):
                st.json(input_data.to_dict(orient="records")[0])

        st.markdown("<hr style='border:1px solid #44444E;'>", unsafe_allow_html=True)