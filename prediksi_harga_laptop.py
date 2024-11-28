import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import LabelEncoder

# Memuat model prediksi
model = pickle.load(open('model.sav', 'rb'))

st.title('Prediksi Harga Laptop')

# Coba menggunakan encoding 'latin1'
df1 = pd.read_csv('laptop_price.csv', encoding='latin1')

# Menggunakan LabelEncoder untuk kolom kategori
label_encoder = LabelEncoder()
df1['Product'] = label_encoder.fit_transform(df1['Product'])
df1['Cpu'] = label_encoder.fit_transform(df1['Cpu'])
df1['Gpu'] = label_encoder.fit_transform(df1['Gpu'])
df1['OpSys'] = label_encoder.fit_transform(df1['OpSys'])

# Fungsi untuk halaman Deskripsi
def show_deskripsi():
    st.write("Selamat datang di aplikasi prediksi harga laptop berbasis web.")
    st.write("<div style='text-align: justify;'>Aplikasi ini menggunakan teknologi <i>Machine Learning</i> untuk memprediksi harga laptop berdasarkan beberapa parameter kunci yang relevan. Dengan memasukkan data seperti <b>Product</b>, <b>Model</b>, <b>Processor</b>, <b>RAM</b>, <b>Memory</b>, <b>Gpu</b>, dan <b>OpSys</b>, pengguna dapat dengan mudah mendapatkan perkiraan harga laptop yang sesuai dengan spesifikasi tersebut. Model prediksi ini dilatih menggunakan data historis yang mencakup ribuan entri, memastikan akurasi dan keandalan hasil prediksi. Aplikasi ini dirancang untuk membantu pengguna membuat keputusan yang lebih bijak, baik untuk pembelian laptop baru maupun bekas. Dengan antarmuka yang sederhana dan mudah digunakan, aplikasi ini cocok untuk konsumen individu maupun penjual yang ingin memperkirakan harga pasar secara cepat dan efisien. Dengan integrasi teknologi canggih dan data yang luas, aplikasi ini memberikan wawasan yang berguna untuk semua kalangan.</div>", unsafe_allow_html=True)
    st.write("Sumber data: https://www.kaggle.com/code/ahmedayad20/laptop-price")
    st.write("Dibuat oleh Rahmanda Putri Radisa - 2024")

# Fungsi untuk halaman Dataset
def show_dataset():
    st.header("Dataset")
    st.dataframe(df1)
    st.markdown("""
( 1 ) **Brand**
   - Merek laptop yang tersedia di pasaran.
  \n(
2 ) **Model**
   - Nama atau tipe spesifik dari laptop.
  \n(
3 ) **Processor**
   - Jenis prosesor yang digunakan pada laptop, misalnya Intel Core i5, AMD Ryzen.
  \n(
4 ) **RAM**
   - Kapasitas memori (RAM) laptop, misalnya 4 GB, 8 GB, 16 GB.
  \n(
5 ) **Memory**
   - Jenis dan kapasitas penyimpanan, misalnya SSD 512 GB atau HDD 1 TB.
  \n(
6 ) **Gpu**
   - Ukuran layar laptop dalam inci, misalnya 13 inci, 15 inci.
  \n(
7 ) **Operating System**
   - Sistem operasi yang terinstal di laptop, misalnya Windows 10, macOS, Linux.
""")

# Fungsi untuk halaman Grafik
def show_grafik():
    st.header("Grafik")
    
    # Pastikan kolom 'Product' ada dalam dataframe
    if 'Product' in df1.columns:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Product", "Cpu", "Ram", "Memory", "Gpu", "ScreenResolution", "OpSys"])
        
        with tab1:
            st.write("Grafik Product")
            chart_product = pd.DataFrame(df1, columns=["Product"])
            st.bar_chart(chart_product['Product'].value_counts())  

        with tab2:
            st.write("Grafik Cpu")
            chart_cpu = pd.DataFrame(df1, columns=["Cpu"])
            st.bar_chart(chart_cpu['Cpu'].value_counts())

        with tab3:
            st.write("Grafik Ram")
            chart_ram = pd.DataFrame(df1, columns=["Ram"])
            st.bar_chart(chart_ram['Ram'].value_counts())

        with tab4:
            st.write("Grafik Memory")
            chart_Memory = pd.DataFrame(df1, columns=["Memory"])
            st.bar_chart(chart_Memory['Memory'].value_counts())

        with tab5:
            st.write("Grafik Gpu")
            chart_screensize = pd.DataFrame(df1, columns=["Gpu"])
            st.bar_chart(chart_screensize['Gpu'].value_counts())

        with tab6:
            st.write("Grafik ScreenResolution")
            chart_screensize = pd.DataFrame(df1, columns=["ScreenResolution"])
            st.bar_chart(chart_screensize['ScreenResolution'].value_counts())

        with tab7:
            st.write("Grafik OpSys")
            chart_screensize = pd.DataFrame(df1, columns=["OpSys"])
            st.bar_chart(chart_screensize['OpSys'].value_counts())
    else:
        st.error("Kolom 'Product' tidak ditemukan dalam dataset.")

# Fungsi untuk halaman Prediksi
def show_prediksi():
    st.header("Prediksi Harga Laptop")
    st.write("Masukkan spesifikasi laptop untuk memprediksi harga:")

    # Input untuk spesifikasi laptop
    Cpu = st.selectbox('Cpu', df1['Cpu'].unique())
    Ram = st.slider('Ram (GB):', 4, 64, 8)

    # Prediksi harga laptop berdasarkan input
    if st.button('Prediksi'):
        # Input data hanya sesuai fitur yang digunakan
        input_data = [[Cpu, Ram]]
        harga_prediksi = model.predict(input_data)  # Prediksi dengan model
        st.write(f'Perkiraan harga laptop: Rp {harga_prediksi[0]:,.2f}')

# Menampilkan menu di sidebar
add_selectbox = st.sidebar.selectbox(
    "PILIH MENU",
    ("Deskripsi", "Dataset", "Grafik", "Prediksi")
)

# Kondisi untuk menampilkan halaman sesuai pilihan
if add_selectbox == "Deskripsi":
    show_deskripsi()
elif add_selectbox == "Dataset":
    show_dataset()
elif add_selectbox == "Grafik":
    show_grafik()
elif add_selectbox == "Prediksi":
    show_prediksi()
