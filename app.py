import numpy as np
from flask import Flask, request, jsonify, render_template
import json
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

# db = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",
#     database="rekomendasi"
# )
#
# cursor = db.cursor()
#
# #Menggabungkan kolom
# query1 = 'SELECT place_id, name, merged_address from places_of_all2'
# query2 = 'SELECT place_id, place_name, rating_review from reviews'
# cursor.execute(query1)
#
# df_all = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
# unique_values = df_all['merged_address'].unique()
#
# cursor.execute(query2)
# df_reviews = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
# cursor.close()
# db.close()


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


    return top10_indices

def predict_rating(user_id, num_places, id_place_list):
    user_id = user_id
    num_places = num_places

    #buat array isi user 2
    user_1 = np.array([user_id for i in range(len(id_place_list))])

    #melakukan prediksi
    pred = rating_model.predict([user_1, id_place_list]).reshape(num_places)

    max_val_rating = 5.0
    min_val_rating = 1.0

    #Ambil top 10 nilai id yang diprediksi
    top_10_ids = (-pred).argsort()[:10]
    #ambil 10 id teratas dari place list
    top_10_places_id = id_place_list[top_10_ids]
    #ambil 5 nilai rating teratas dari place list
    #Mengalikan dengan rumus min-max lagi agar memiliki nilai yang sama dengan ketika diawal sebelum dilakukan prediksi
    top_10_places_rating = pred[top_10_ids]*(max_val_rating - min_val_rating) + min_val_rating

    result_dict = {
    'top_10_places_id': top_10_places_id.tolist(),  # Konversi ke list jika belum
    'top_10_places_rating': top_10_places_rating.round(1).tolist()  # Konversi ke list jika belum
    }

    json_result = json.dumps(result_dict, ensure_ascii=False)
    return json_result



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

