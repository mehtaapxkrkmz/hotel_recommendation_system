import streamlit as st
import pandas as pd

# Veri setini yükleyelim
df = pd.read_csv("C:/Users/Asus/.cache/kagglehub/datasets/jiashenliu/515k-hotel-reviews-data-in-europe/versions/1/Hotel_Reviews.csv")

# 'Review_Date' sütununu datetime formatına dönüştür
df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')

# Yıl ve Ay bilgisi ekleyelim
df['Review_Year'] = df['Review_Date'].dt.year
df['Review_Month'] = df['Review_Date'].dt.month
df['Review_Month_Name'] = df['Review_Date'].dt.strftime('%B')

# Mevsim fonksiyonu
def season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    elif month in [12, 1, 2]:
        return 'Winter'
    else:
        return "Unknown!"

# Mevsim sütunu ekleyelim
df['Season'] = df['Review_Date'].dt.month.apply(season)

# Olumlu ve olumsuz yorumları birleştirerek 'Review_Text' sütununu oluşturuyoruz
df['Review_Text'] = df['Positive_Review'] + " " + df['Negative_Review']

# 'Hotel_Address' sütunundan ülke bilgisini alalım
df['Country'] = df['Hotel_Address'].str.split(' ').str[-1]

# Streamlit arayüzü başlat
st.title("Hotel Reviews and Seasonal Analysis")

# Kullanıcıdan yıl seçimini alalım
selected_year = st.selectbox('Choose a year', df['Review_Year'].unique())

# Seçilen yıl için ayları filtreleyelim
selected_month = st.selectbox('Choose a month', df[df['Review_Year'] == selected_year]['Review_Month_Name'].unique())

# Kullanıcıdan puan seçimini alalım.
min_score = st.slider('Minimum Score', 1, 10, 1)
max_score = st.slider('Maximum Score', 1, 10, 10)

# Kullanıcıdan ülke seçimini alalım
selected_country = st.selectbox('Choose a country', df['Country'].unique())

# Seçilen yıl, ay, puan aralığı ve ülke bilgisiyle filtreleme yapalım
filtered_data = df[
    (df['Review_Year'] == selected_year) & 
    (df['Review_Month_Name'] == selected_month) &
    (df['Reviewer_Score'] >= min_score) &
    (df['Reviewer_Score'] <= max_score) &
    (df['Country'] == selected_country)
]

# Mevsim bilgisi de dahil olmak üzere veriyi gösterecek şekilde güncelleyelim
st.write("Hotel reviews and seasonal analysis for the selected filters:", 
         filtered_data[['Hotel_Address', 'Review_Date', 'Review_Month_Name', 'Season', 'Review_Text', 'Reviewer_Score', 'Country']])

# Mevsime göre yorum sayılarının dağılımını gösterelim
season_counts = df['Season'].value_counts()
st.write("Number of Comments by Season:")
st.bar_chart(season_counts)

# Ekstra: Kullanıcı puanı ile otel puanı arasındaki farkı analiz et
df['Score_Difference'] = df['Reviewer_Score'] - df['Average_Score']
st.write("Score Difference Analysis", df[['Reviewer_Score', 'Average_Score', 'Score_Difference']].head())
