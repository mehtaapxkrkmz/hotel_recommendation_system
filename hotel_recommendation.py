import pandas as pd  
import streamlit as st  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.neighbors import NearestNeighbors  

# Veri setini yükleyelim  
df = pd.read_csv("C:/Users/Asus/.cache/kagglehub/datasets/jiashenliu/515k-hotel-reviews-data-in-europe/versions/1/Hotel_Reviews.csv")  

st.title("Hotel Reviews and Seasonal Analysis")  

# 'Review_Date' sütununu datetime formatına dönüştür  
df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')  
df.dropna(subset=['Review_Date'], inplace=True)  # NaT değerlerini temizle  

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
    return "Unknown!"  

# Mevsim sütunu ekleyelim  
df['Season'] = df['Review_Date'].dt.month.apply(season)  

# Olumlu ve olumsuz yorumları birleştirerek 'Review_Text' sütununu oluşturuyoruz  
df['Review_Text'] = df['Positive_Review'] + " " + df['Negative_Review']  

# 'Hotel_Address' sütunundan ülke bilgisini alalım  
df['Country'] = df['Hotel_Address'].str.split(' ').str[-1]  

# Otel öneri sistemi  
st.subheader("Hotel Recommendation System")  

# TF-IDF vektörleştirme ve model eğitimi  
vectorizer = TfidfVectorizer(max_features=500)  
X_reviews = vectorizer.fit_transform(df['Review_Text']).toarray()  
X_features = pd.concat([pd.DataFrame(X_reviews), df[['Review_Year', 'Reviewer_Score']].reset_index(drop=True)], axis=1)  
X_features.columns = X_features.columns.astype(str)  

knn = NearestNeighbors(n_neighbors=10, metric='cosine')  
knn.fit(X_features)  

# Sidebar filtreleme seçenekleri  
st.sidebar.subheader("Filter Options")  

# Ülke seçimi  
selected_country = st.sidebar.selectbox('Choose a country', df['Country'].unique())  
st.write(f"Selected Country: {selected_country}")  

# Yıl seçimi  
selected_year = st.sidebar.selectbox('Choose a year', df['Review_Year'].unique())  
st.write(f"Selected Year: {selected_year}")  

# Ay seçimi  
selected_month = st.sidebar.selectbox('Choose a month', df['Review_Month_Name'].unique())  
st.write(f"Selected Month: {selected_month}")  

# Minimum ve maksimum skor seçimi  
min_score = st.sidebar.slider('Select Minimum Score', min_value=0, max_value=10, value=0)  
max_score = st.sidebar.slider('Select Maximum Score', min_value=0, max_value=10, value=10)  
st.write(f"Selected Score Range: {min_score} - {max_score}")  

# Filtreleme işlemi  
filtered_df = df[(df['Country'] == selected_country) &   
                 (df['Review_Year'] == selected_year) &   
                 (df['Review_Month_Name'] == selected_month) &   
                 (df['Reviewer_Score'] >= min_score) &   
                 (df['Reviewer_Score'] <= max_score)]  

# Uygun otelleri önerelim  
st.write(f"Filtered Data for {selected_country} in {selected_year}, {selected_month} with scores between {min_score} and {max_score}:")  
st.write(filtered_df[['Hotel_Name', 'Reviewer_Score', 'Review_Text', 'Country', 'Review_Year', 'Review_Month_Name']])  

# Kullanıcıya uygun otelleri önerme
# Kullanıcıya uygun otelleri önerme
if st.button("Recommend Hotels"):
    if not filtered_df.empty:
        # En yakın otelleri bulalım
        filtered_reviews = vectorizer.transform(filtered_df['Review_Text']).toarray()
        filtered_features = pd.concat([pd.DataFrame(filtered_reviews), filtered_df[['Review_Year', 'Reviewer_Score']].reset_index(drop=True)], axis=1)
        filtered_features.columns = filtered_features.columns.astype(str)
        
        # Seçilen bir yorumla en yakın otelleri bulma
        distances, indices = knn.kneighbors(filtered_features, n_neighbors=10)  # Burada n_neighbors=10 daha fazla otel önerir
        
        # Önerilen otelleri listele
        recommended_hotels = []
        for idx_list in indices:
            for idx in idx_list:
                if idx < len(filtered_df):  # Geçerli bir indeks olup olmadığını kontrol et
                    recommended_hotels.append(filtered_df.iloc[idx])

        if recommended_hotels:
            st.write("Recommended Hotels based on the filters:")
            st.write(pd.DataFrame(recommended_hotels)[['Hotel_Name', 'Reviewer_Score', 'Review_Text', 'Country', 'Review_Year', 'Review_Month_Name']])
        else:
            st.write("No valid hotels found based on the selected criteria.")
    else:
        st.write("No hotels match the selected filters.")
