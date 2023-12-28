import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
import time
import joblib

st.set_option('deprecation.showPyplotGlobalUse', False)
# Fungsi untuk memuat model
def load_model():
    # Ganti dengan lokasi penyimpanan model Anda
    model = joblib.load('model.pkl')
    return model

# Fungsi untuk melakukan prediksi berdasarkan data input
def predict_loan_status(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Fungsi untuk menampilkan visualisasi data
def show_data_visualization():
    df = pd.read_csv('data_clean.csv')

    # Memilih kolom-kolom yang diinginkan
    selected_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status']
    data_selected = df[selected_columns]

    # Mendefinisikan pemetaan label ke nilai numerik
    label_mapping = {'Y': 1, 'N': 0}

    # Mengganti nilai kategori dengan nilai numerik menggunakan map
    data_selected['Loan_Status'].map(label_mapping)

    # Mengisi missing value pada 'LoanAmount' dengan nilai rata-rata
    mean_loan_amount = data_selected['LoanAmount'].mean()
    data_selected['LoanAmount'].fillna(mean_loan_amount, inplace=True)

    # Mengisi missing value pada 'Loan_Amount_Term' dengan nilai modus
    mode_loan_term = data_selected['Loan_Amount_Term'].mode()[0]
    data_selected['Loan_Amount_Term'].fillna(mode_loan_term, inplace=True)

    # Mengisi missing value pada 'Credit_History' dengan nilai modus
    mode_credit_history = data_selected['Credit_History'].mode()[0]
    data_selected['Credit_History'].fillna(mode_credit_history, inplace=True)


    # Visualisasi
    st.title('Visualisasi Dataset')

    st.write('Preview Dataset')
    st.write(df.head(10))
    
    st.subheader('Histogram')
    data_selected.hist(figsize=(10, 8))
    plt.tight_layout()
    st.pyplot()

    st.subheader('Heatmap Korelasi')
    correlation_matrix = data_selected.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()

    st.subheader('Count Plot Status Pinjaman')
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Loan_Status', data=df)
    st.pyplot()

def main():
    # Muat model
    model = load_model()

    st.title('Prediksi Status Pinjaman')

    # Pilihan untuk visualisasi atau prediksi
    option = st.sidebar.radio('Pilih Fitur', ['Visualisasi Dataset', 'Prediksi Peminjam'])

    if option == 'Visualisasi Dataset':
        # Tampilkan visualisasi data
        show_data_visualization()
    elif option == 'Prediksi Peminjam':
        df = pd.read_csv('data_clean.csv')  # Baca kembali DataFrame
        # Judul untuk input data prediksi
        st.subheader('Masukkan Data Peminjam')

        # Contoh input untuk data ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
        applicant_income = int(st.number_input('Pendapatan Utama Peminjam', format='%d', value=0))
        coapplicant_income = int(st.number_input('Pendapatan Tambahan Peminjam', format='%d', value=0))
        loan_amount = int(st.number_input('Jumlah Pinjaman yang Diajukan', format='%d', value=0))
        loan_amount_term_options = df['Loan_Amount_Term'].unique()
        loan_amount_term = st.selectbox('Jangka Waktu Pinjaman yang Diajukan', loan_amount_term_options)
        df['Credit_History_Text'] = df['Credit_History'].apply(lambda x: 'Ya' if x == 0 else 'Tidak')
        credit_history_options = df['Credit_History_Text'].unique()
        credit_history = st.selectbox('Apakah Peminjam Memiliki Riwayat Kredit yang Buruk?', credit_history_options)


        # Prediksi saat tombol ditekan
        if st.button('Predict'):
            # Proses data input sebelum diprediksi
            input_data = np.array([applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history]).reshape(1, -1)
            # Lakukan prediksi
            prediction = predict_loan_status(model, input_data)
            
            # Tampilkan hasil prediksi
            if prediction[0] == 0:
                st.write('Hasil Prediksi: Peminjam Tidak Layak Diberi Pinjaman')
            else:
                st.write('Hasil Prediksi: Peminjam Layak Diberi Pinjaman')

if __name__ == '__main__':
    main()
