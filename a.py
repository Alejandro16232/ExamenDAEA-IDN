import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Cargar el archivo CSV
df = pd.read_csv('EdX.csv')

# Descargar recursos de NLTK necesarios
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar el lematizador y las stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Función para preprocesar el texto
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]

# Preprocesar el texto de la columna 'Course Description'
df['Processed Description'] = df['Course Description'].apply(preprocess)

# Combinar los tokens en un solo string por descripción para aplicar TF-IDF
df['Processed Description'] = df['Processed Description'].apply(lambda x: ' '.join(x))

# Generar la matriz TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Processed Description'])

# Función para obtener palabras clave y su similitud
def get_keywords_with_similarity(reference_index, top_n=10):
    # Calcular la matriz de similitud de coseno
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Obtener la similitud con el texto de referencia
    similarity_scores = similarity_matrix[reference_index]
    
    # Calcular la relevancia de cada palabra en base a la similitud
    feature_names = vectorizer.get_feature_names_out()
    keywords_scores = {}
    
    for word_index in range(len(feature_names)):
        word = feature_names[word_index]
        word_score = tfidf_matrix[:, word_index].toarray().flatten()
        similarity_sum = np.dot(word_score, similarity_scores)
        keywords_scores[word] = similarity_sum
    
    # Ordenar las palabras clave por relevancia
    sorted_keywords = sorted(keywords_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Devolver las top_n palabras clave más similares
    return sorted_keywords[:top_n]

# Obtener las 10 palabras clave más similares al texto en el índice 0
keywords_with_similarity = get_keywords_with_similarity(0)

# Convertir el resultado a JSON para una mejor visualización
result = {word: round(similarity, 2) for word, similarity in keywords_with_similarity}
print(json.dumps(result, indent=4))
