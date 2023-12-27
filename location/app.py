import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
from sklearn.metrics.pairwise import linear_kernel

# Create flask app
flask_app = Flask(__name__)

recommender_model = joblib.load('models/tfidf_vectorizer.joblib')
matrix_all = joblib.load('models/tfidf_matrix.joblib')
rating_model = tf.keras.models.load_model('models/tensorflow_model.h5')
df_all_places = pd.read_csv('dataset/new-fix-data.csv')
df_recom = df_all_places.copy()

selected_columns = df_recom[['user_id','rating_review','place_id', 'name', 'address_city', 'mbti_labels']]

#gabung string daerah dan nama objek wisata
selected_columns['merge_name_address'] = selected_columns['name'] + ' ' + selected_columns['address_city']

unique_values = selected_columns['merge_name_address'].unique()

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
        'merge_name_address': unique_values[top10_indices],
        'cosine_similarity': similarity_scores[top10_indices]
    })

    # Cocokkan hasil rekomendasi dengan DataFrame utama
    result_df = pd.merge(top10_recommendations, selected_columns[['merge_name_address', 'place_id']], on='merge_name_address')

    # Hapus duplikat berdasarkan D'place_id'
    result_df = result_df.drop_duplicates(subset='place_id')

    return result_df

def predict_rating(user_id, mbti, recommendations):
    num_places = 10

    id_place_list = recommendations['place_id'].values

    #buat array isi user 2
    user_1 = np.array([mbti for i in range(len(id_place_list))])

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
        'User ID': [user_id] * len(top_10_places_id),
        'places_id': top_10_places_id.tolist(),  # Konversi ke list jika belum
        'places_rating': top_10_places_rating.round(1).tolist(),  # Konversi ke list jika belum
        'mbti_id': [mbti] * len(top_10_places_id)
    }

    return result_dict


def convert_mbti(result):
    mbti_dict = {'ENTJ': 1, 'INTJ': 2, 'INFJ': 3, 'ENFJ': 4, 'ESTJ': 5, 'ISTJ': 6,
                 'ISFJ': 7, 'ESFJ': 8, 'ENTP': 9, 'INTP': 10, 'INFP': 11, 'ENFP': 12,
                 'ESTP': 13, 'ISTP': 14, 'ISFP': 15, 'ESFP': 16}

    return mbti_dict.get(result, None)


@flask_app.route("/")
def Home():
    return {"health_check": "NGENE TO?", "model_version": "OKE?"}

@flask_app.route("/predict", methods=["GET"])
def predict():
    # Get the values of 'daerah', 'object', 'mbti', and 'user' from the query string
    daerah = request.args.get('daerah')
    obj = request.args.get('object')
    mbti = request.args.get('mbti')
    user = request.args.get('user')

    # Check if any of the required parameters is missing
    if None in [daerah, obj, mbti, user]:
        return jsonify({"error": "Missing required parameters"})

    mbti_convert = convert_mbti(mbti)

    print()
    recommendation = recommend_top10(daerah, obj, recommender_model, matrix_all)
    print(recommendation)
    result_df = predict_rating(user, mbti_convert, recommendation)

    print(result_df)
    return jsonify({"top_10": result_df})

if __name__ == "__main__":
    flask_app.run(debug=True, port=5009)
