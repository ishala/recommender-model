import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics.pairwise import linear_kernel
import mysql.connector

# Create flask app
flask_app = Flask(__name__)

recommender_model = joblib.load('models/tfidf_vectorizer.joblib')
matrix_all = joblib.load('models/tfidf_matrix.joblib')
rating_model = tf.keras.models.load_model('models/tensorflow_model.h5')

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="rekomendasi"
)

cursor = db.cursor()

#Menggabungkan kolom
query1 = 'SELECT place_id, name, merged_address from places_of_all2'
query2 = 'SELECT place_id, place_name, rating_review from reviews'
cursor.execute(query1)

df_all = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
unique_values = df_all['merged_address'].unique()

cursor.execute(query2)
df_reviews = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
cursor.close()
db.close()

def recommend_top10(daerah, nama_objek, tfidf, tfidf_matrix):
    # Gabungkan daerah dan nama_objek untuk mencari padanan di nilai unik 'merge_name_address'
    input_text = f"{nama_objek} {daerah}"

    # Transform input_text menggunakan TF-IDF yang sudah di-fit sebelumnya
    input_tfidf = tfidf.transform([input_text])

    # Hitung cosine similarity antara input_text dan nilai unik
    similarity_scores = linear_kernel(input_tfidf, tfidf_matrix).flatten()

    # Urutkan skor similarity dari tertinggi ke terendah
    sorted_scores_indices = similarity_scores.argsort()[::-1]

    # Ambil top 10 rekomendasi (exluding input_text itself)
    top10_indices = sorted_scores_indices[1:11]

    top10_recommendations = pd.DataFrame({
        'merged_address': unique_values[top10_indices],
        'cosine_similarity': similarity_scores[top10_indices]
    })

    # Cocokkan hasil rekomendasi dengan DataFrame utama
    result_df = pd.merge(top10_recommendations, df_all[['merged_address', 'place_id']], on='merged_address')

    # Hapus duplikat berdasarkan 'place_id'
    result_df = result_df.drop_duplicates(subset='place_id')

    return result_df

def predict_rating(user_id, num_places, recommendations):
    user_id = user_id
    num_places = num_places

    #Ambil sample sebanyak nilai deklarasi
    place_list = recommendations['place_id'].values
    #buat array isi user 2
    user_1 = np.array([user_id for i in range(len(place_list))])

    #melakukan prediksi
    pred = rating_model.predict([user_1, place_list]).reshape(num_places)

    #Ambil top 5 nilai id yang diprediksi
    top_5_ids = (-pred).argsort()[:5]
    #ambil 5 id teratas dari place list
    top_5_places_id = place_list[top_5_ids]
    #ambil 5 nilai rating teratas dari place list
    #Mengalikan dengan rumus min-max lagi agar memiliki nilai yang sama dengan ketika diawal sebelum dilakukan prediksi
    top_5_places_rating = pred[top_5_ids]*(df_all['rating_review'].max() - df_reviews['rating_review'].min()) + df_reviews['rating_review'].min()

    # Mengambil place_id dan nama_tempat dari lima nilai teratas
    top_5_places_info = df_all[df_all['place_id'].isin(top_5_places_id)][['place_id', 'name']].drop_duplicates()

    # Membuat DataFrame hasil
    result_df = pd.DataFrame({
        'User ID': [user_id] * len(top_5_places_id),
        'Place ID': top_5_places_id,
        'Nama Tempat': top_5_places_info['name'].values,
        'Prediksi Rating': top_5_places_rating.round(1)
    })

    return result_df



@flask_app.route("/")
def Home():
    return render_template("input.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    values_input = [x for x in request.form.values()]

    recommendation = recommend_top10(values_input[0], values_input[1], recommender_model, matrix_all)
    result_df = predict_rating(1, 10, recommendation)

    return render_template("input.html", prediction_text = "Ini prediksinya {}".format(result_df))

if __name__ == "__main__":
    flask_app.run(debug=True)

