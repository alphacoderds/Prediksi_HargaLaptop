import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Memuat model prediksi
model = pickle.load(open('model.sav', 'rb'))

st.title('Prediksi Harga Laptop')

# Coba menggunakan encoding 'latin1'
df1 = pd.read_csv('laptop_price.csv', encoding='latin1')

# Fungsi untuk halaman Deskripsi
def show_deskripsi():
    st.write("Selamat datang di aplikasi prediksi harga laptop berbasis web.")
    st.write("<div style='text-align: justify;'>Aplikasi ini menggunakan teknologi <i>Machine Learning</i> untuk memprediksi harga laptop berdasarkan beberapa parameter kunci yang relevan. Dengan memasukkan data seperti <b>Product</b>, <b>Model</b>, <b>Processor</b>, <b>RAM</b>, <b>Storage</b>, <b>Screen Size</b>, dan <b>OpSys</b>, pengguna dapat dengan mudah mendapatkan perkiraan harga laptop yang sesuai dengan spesifikasi tersebut. Model prediksi ini dilatih menggunakan data historis yang mencakup ribuan entri, memastikan akurasi dan keandalan hasil prediksi. Aplikasi ini dirancang untuk membantu pengguna membuat keputusan yang lebih bijak, baik untuk pembelian laptop baru maupun bekas. Dengan antarmuka yang sederhana dan mudah digunakan, aplikasi ini cocok untuk konsumen individu maupun penjual yang ingin memperkirakan harga pasar secara cepat dan efisien. Dengan integrasi teknologi canggih dan data yang luas, aplikasi ini memberikan wawasan yang berguna untuk semua kalangan.</div>", unsafe_allow_html=True)
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
5 ) **Storage**
   - Jenis dan kapasitas penyimpanan, misalnya SSD 512 GB atau HDD 1 TB.
  \n(
6 ) **Screen Size**
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Product", "cpu", "RAM", "Storage", "Screen Size"])
        
        with tab1:
            st.write("Grafik Product")
            chart_product = pd.DataFrame(df1, columns=["Product"])
            st.bar_chart(chart_product['Product'].value_counts())  # Pastikan 'Product' ada

        with tab2:
            st.write("Grafik cpu")
            chart_cpu = pd.DataFrame(df1, columns=["cpu"])
            st.bar_chart(chart_cpu['cpu'].value_counts())

        with tab3:
            st.write("Grafik RAM")
            chart_ram = pd.DataFrame(df1, columns=["RAM"])
            st.bar_chart(chart_ram['RAM'].value_counts())

        with tab4:
            st.write("Grafik Storage")
            chart_storage = pd.DataFrame(df1, columns=["Storage"])
            st.bar_chart(chart_storage['Storage'].value_counts())

        with tab5:
            st.write("Grafik Screen Size")
            chart_screensize = pd.DataFrame(df1, columns=["Screen Size"])
            st.bar_chart(chart_screensize['Screen Size'].value_counts())
    else:
        st.error("Kolom 'Product' tidak ditemukan dalam dataset.")

# Fungsi untuk halaman Prediksi
def show_prediksi():
    st.header("Prediksi Harga Laptop")
    st.write("Masukkan spesifikasi laptop untuk memprediksi harga:")

    # Input untuk spesifikasi laptop
    product = st.selectbox('Product', df1['Product'].unique())  #
    cpu = st.selectbox('Cpu', df1['Cpu'].unique())
    ram = st.slider('RAM (GB):', 4, 64, 8)
    storage = st.slider('Storage (GB):', 128, 2048, 512)
    screen_size = st.slider('Screen Size (inch):', 10, 20, 15)
    OpSys = st.selectbox('OpSys', df1['OpSys'].unique()) 


    # Prediksi harga laptop berdasarkan input
    if st.button('Prediksi'):
        input_data = [[product, cpu, ram, storage, screen_size, OpSys]]
        harga_prediksi = model.predict(input_data)
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
