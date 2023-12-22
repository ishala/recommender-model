import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib


#FUNGSI
def recommend_top10(daerah, nama_objek):
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

    # Hapus duplikat berdasarkan 'place_id'
    result_df = result_df.drop_duplicates(subset='place_id')

    return result_df

def predict_rating(user_id, mbti_labels, num_places=10):
    num_places = num_places

    #Ambil sample sebanyak nilai deklarasi
    place_list = recommendations['place_id'].values
    #buat array isi user 2
    user_1 = np.array([mbti_labels for i in range(len(place_list))])

    #melakukan prediksi
    pred = model.predict([user_1, place_list]).reshape(num_places)

    #Ambil top 5 nilai id yang diprediksi
    top_5_ids = (-pred).argsort()[:5]
    #ambil 5 id teratas dari place list
    top_5_places_id = place_list[top_5_ids]
    #ambil 5 nilai rating teratas dari place list
    #Mengalikan dengan rumus min-max lagi agar memiliki nilai yang sama dengan ketika diawal sebelum dilakukan prediksi
    top_5_places_rating = pred[top_5_ids]*(selected_columns['rating_review'].max() - selected_columns['rating_review'].min()) + selected_columns['rating_review'].min()

    # Mengambil place_id dan nama_tempat dari lima nilai teratas
    top_5_places_info = selected_columns[selected_columns['place_id'].isin(top_5_places_id)][['place_id', 'name']].drop_duplicates()

    # Membuat DataFrame hasil
    result_df = pd.DataFrame({
        'User ID': [user_id] * len(top_5_places_id),
        'Place ID': top_5_places_id,
        'Nama Tempat': top_5_places_info['name'].values,
        'Prediksi Rating': top_5_places_rating.round(1),
        'MBTI': [mbti_labels] * len(top_5_places_id)
    })

    return result_df

# db = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",
#     database="rekomendasi"
# )
#
# cursor = db.cursor()
#
# #Ambil dataset
# query1 = 'SELECT * from places_of_all2'
# query2 = 'SELECT * from reviews'
#
# cursor.execute(query1)
# df_all_places = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
#
# cursor.execute(query2)
# df_review = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
# cursor.close()
# db.close()

df_all_places = pd.read_csv('dataset/places-of-all2.csv')
df_review = pd.read_csv('dataset/review-fix.csv')


#Menggabungkan kolom
df_review['mbti_labels'] = np.random.randint(1, 17, size=len(df_review))

df_recom = pd.merge(df_all_places, df_review, on='place_id', how='inner')
print(df_recom.columns)

#Ambil kolom yang digunakan
selected_columns = df_recom[['user_id','rating_review','place_id', 'name', 'address_city', 'mbti_labels']]

#gabung string daerah dan nama objek wisata
selected_columns['merge_name_address'] = selected_columns['name'] + ' ' + selected_columns['address_city']

#Buat vectorizer
tfidf = TfidfVectorizer()
unique_values = selected_columns['merge_name_address'].unique()

#fit transform matrix unique values
tfidf_matrix = tfidf.fit_transform(unique_values)


#Tes
daerah_input = "Solo"
nama_objek_input = "Taman"

recommendations = recommend_top10(daerah_input, nama_objek_input)


#GENERATE RATING
train_data, valid_data = train_test_split(selected_columns, test_size=0.2,random_state=42)
dim_places = selected_columns['place_id'].max() + 1
dim_users = selected_columns['mbti_labels'].max() + 1

#arsitektur place
place_input = Input(shape=(1,), name="Place_Input")
place_embedding = Embedding(dim_places, 16, embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6), name="Place_Embedd")(place_input)
place_vector = Flatten(name="Flatten_Places")(place_embedding)

#arsitektur user
user_input = Input(shape=(1,), name="User_Input")
user_embedding = Embedding(dim_users, 16, embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6), name="User_Embedding")(user_input)
user_vector = Flatten(name="Flatten_Users")(user_embedding)

#Dot Produk
prod = Dot(name="Dot_Product", axes=1)([place_vector, user_vector])

#Penambahan Dense
dense = Dense(1,activation='relu')(prod)

model = Model([user_input, place_input], dense)
model.compile(loss='mean_squared_error',optimizer='adam')

print(model.summary())


#TRAINING MODEL
#Buat fungsi callback
callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

y_train = (train_data['rating_review'] - train_data['rating_review'].min()) / (train_data['rating_review'].max() - train_data['rating_review'].min())
y_val=(valid_data['rating_review'] - valid_data['rating_review'].min()) / (valid_data['rating_review'].max() - valid_data['rating_review'].min())

history = model.fit(x=[train_data['mbti_labels'],train_data['place_id']],
                    y=y_train,
                    batch_size=64,
                    epochs=10,
                    verbose=1,
                    validation_data=([valid_data['mbti_labels'], valid_data['place_id']], y_val),
                    callbacks = [callback])


#Prediksi rating
result_df = predict_rating(2, 1, 10)

print(result_df)

#Simpan model
# Simpan TF-IDF Vectorizer
joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')
# Simpan Matrix TF-IDF
joblib.dump(tfidf_matrix, 'models/tfidf_matrix.joblib')

#Simpan model Tensorflow
model.save('models/tensorflow_model.h5')

