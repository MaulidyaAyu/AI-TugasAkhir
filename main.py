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
    data = pd.read_csv('loadPredictionAsli.csv')

    # Memilih kolom-kolom yang diinginkan
    selected_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status']

    # Menggunakan metode loc untuk memilih kolom yang diinginkan
    data_selected = data.loc[:, selected_columns]

    # Mendefinisikan pemetaan label ke nilai numerik
    label_mapping = {'Y': 1, 'N': 0}

    # Mengganti nilai kategori dengan nilai numerik menggunakan map
    data_selected['Loan_Status'] = data_selected['Loan_Status'].map(label_mapping)

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
    st.title('Visualisasi Data')
    
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
    sns.countplot(x='Loan_Status', data=data_selected)
    st.pyplot()

def main():
    # Muat model
    model = load_model()

    st.title('Prediksi Status Pinjaman')

    # Pilihan untuk visualisasi atau prediksi
    option = st.sidebar.radio('Pilih', ['Visualisasi Data', 'Prediksi'])

    if option == 'Visualisasi Data':
        # Tampilkan visualisasi data
        show_data_visualization()
    else:
        # Judul untuk input data prediksi
        st.subheader('Masukkan Data untuk Prediksi')

        # Contoh input untuk data ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
        applicant_income = st.number_input('Pendapatan Utama')
        coapplicant_income = st.number_input('Pendapatan Tambahan')
        loan_amount = st.number_input('Jumlah Pinjaman yang Diajukan')
        loan_amount_term = st.selectbox('Jangka Waktu Pinjaman yang Diajukan', options=unique_dates)
        credit_history = st.selectbox('Apakah Anda Memiliki Riwayat Kredit?', options=unique_dates)
        

        # Prediksi saat tombol ditekan
        if st.button('Predict'):
            # Proses data input sebelum diprediksi
            input_data = np.array([applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history]).reshape(1, -1)
            # Lakukan prediksi
            prediction = predict_loan_status(model, input_data)
            
            # Tampilkan hasil prediksi
            if prediction[0] == 0:
                st.write('Hasil Prediksi: Tidak Layak Diberi Pinjaman')
            else:
                st.write('Hasil Prediksi: Layak Diberi Pinjaman')

if __name__ == '__main__':
    main()
