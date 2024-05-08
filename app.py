import pandas as pd
import streamlit as st
import pickle
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# Load your Excel file containing the book data
@st.cache_data
def load_data():
    df = pd.read_excel("merged-armenian-books-dataset.xlsx")
    # Remove duplicates based on 'title' and 'author' combination
    df = df.drop_duplicates(subset=['Title', 'Author'])
    # Select only required columns
    df = df[['Title', 'Author', 'Description', 'Publisher', 'Genre']]
    return df

# Function to preprocess text data
def preprocess_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Streamlit UI for exploring book data
def explore_data(df):
    st.title("Explore Book Data")
    
    # Display the number of books in each genre with colorful bar chart
    genre_counts = df['Genre'].value_counts()
    num_unique_genres = len(genre_counts)
    palette = sns.color_palette("hls", num_unique_genres)
    
    plt.figure(figsize=(10, 6))
    genre_counts.plot(kind='bar', color=palette)
    plt.ylabel('Count')
    plt.title('Genre Distribution')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

    # Allow users to filter books by genre
    selected_genre = st.sidebar.selectbox("Select Genre", df['Genre'].unique())
    filtered_books = df[df['Genre'] == selected_genre]
    
    # Display information about filtered books
    st.subheader(f"Գրքեր '{selected_genre}' ժանրում")
    st.dataframe(filtered_books)


# Streamlit UI
def main():
    # Set page title and favicon
    st.set_page_config(page_title="Armenian Books Genre Classifier", page_icon="📚")

    #Set background color and font
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
            # background-image: url("books_bg.jpg");
            # background-size: cover;
            font-family: Arial, sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.title("Armenian Books Genre Classifier")
    st.image("books_bg.jpg", width = 350)
            
#     st.markdown("""
#     Գրքերի ժանրերը:
# - Ժամանակակից գրականություն              1790
# - Դետեկտիվ և թրիլլեր                     1260
# - Դասական գրականություն                  1054
# - Սիրավեպ                                 638
# - Ֆանտաստիկա                              575
# - Պատմվածք                                414
# - Վեպ                                     264
# - Պոեզիա                                  188
# - Մանկական գրականություն                  148
# - Ոչ գեղարվեստական գրականություն           85
# - Հոգևոր գրականություն                     69
# - Գեղարվեստական գրականություն              45
# - Դրամատուրգիա                             43
# - Նովել                                    40
# - Թատերգություն                            12
# - Գիտական ֆանտաստիկա                        8
# - Արկածային                                 7
# """)

    # Input fields
    st.sidebar.header("Enter Book Details")
    title = st.sidebar.text_input("Title")
    description = st.sidebar.text_area("Description", height=150)
    author = st.sidebar.text_input("Author")
    publisher = st.sidebar.text_input("Publisher")

    # Preprocess data
    text = ' '.join([title, description, author, publisher])
    preprocessed_text = preprocess_text(text)

    # Load the pre-trained CatBoost model
    with open("catboost-model-final.pkl", "rb") as f:
        model = pickle.load(f)

    # Load the pre-trained TfidfVectorizer
    with open("tfidf_vectorizer-final.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    # Make prediction
    if st.sidebar.button("Predict Genre"):
        # Transform the preprocessed text using the loaded TfidfVectorizer
        tfidf_matrix = tfidf_vectorizer.transform([preprocessed_text])
        # Make prediction
        prediction_probs = model.predict_proba(tfidf_matrix)[0]
        
        # Get top 5 predicted classes and their probabilities
        top_classes = model.classes_[prediction_probs.argsort()[::-1][:5]]
        top_probs = prediction_probs[prediction_probs.argsort()[::-1][:5]]
        
        # Print the top 5 predicted classes and their probabilities
        for i in range(len(top_classes)):
            st.success(f"Predicted Genre: {top_classes[i]} ({top_probs[i]*100:.2f}%)")

    # Load book data
    df = load_data()
    
    # Option to switch between pages
    page = st.sidebar.radio("Navigation", ["Book Genre Classifier", "Explore Book Data"])

    # Display selected page
    if page == "Book Genre Classifier":
        pass  # This page is already implemented above
    elif page == "Explore Book Data":
        explore_data(df)

if __name__ == "__main__":
    main()







