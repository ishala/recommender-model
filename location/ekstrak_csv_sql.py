import pandas as pd
import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="rekomendasi"
)

df = pd.read_csv('dataset/cutted_reviews.csv')

data_count_prev = 0
data_count_next = len(df)

cursor = db.cursor()

try:
    for i in range(data_count_prev, data_count_next):
        sql = "INSERT INTO reviews (place_id, place_name, reviewer_name, rating_review, review_text, published_at, published_at_date, review_likes_count, total_number_of_reviews_by_reviewer, total_number_of_photos_by_reviewer, reviewer_url, reviewer_profile_picture, review_photos, user_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        sql_insert = (
            int(df['place_id'][i]), df['place_name'][i], df['reviewer_name'][i],
            float(df['rating_review'][i]), df['review_text'][i], df['published_at'][i],
            df['published_at_date'][i], int(df['review_likes_count'][i]),
            int(df['total_number_of_reviews_by_reviewer'][i]), int(df['total_number_of_photos_by_reviewer'][i]), df['reviewer_url'][i],
            df['reviewer_profile_picture'][i], df['review_photos'][i], int(df['user_id'][i])
        )







        # sql = "INSERT INTO places_of_all2 (place_id, name, description, reviews, owner_name, featured_image, main_category, rating, workday_timing, closed_on, phone, address, link, coordinates, address_ward, address_street, address_city, address_postal_code, address_state, address_country_code, data_id, accessibility_enabled, hours, most_popular_times, popular_times, planning_enabled, children_enabled, main_category_as_labels) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        # sql_insert = (
        #     int(df['place_id'][i]), df['name'][i], df['description'][i],
        #     int(df['reviews'][i]), df['owner_name'][i], df['featured_image'][i],
        #     df['main_category'][i], float(df['rating_place'][i]),
        #     df['workday_timing'][i], df['closed_on'][i], df['phone'][i],
        #     df['address'][i], df['link'][i], df['coordinates'][i], df['address_ward'][i], df['address_street'][i],
        #     df['address_city'][i], int(df['address_postal_code'][i]), df['address_state'][i], df['address_country_code'][i],
        #     df['data_id'][i], df['accessibility_enabled'][i], df['hours'][i], df['most_popular_times'][i], df['popular_times'][i],
        #     df['planning_enabled'][i],df['children_enabled'][i], int(df['main_category_as_labels'][i])
        # )

        cursor.execute(sql, sql_insert)

        # Commit every 100 iterations (you can adjust this value)
        if i % 100 == 0:
            db.commit()

    # Commit any remaining changes
    db.commit()

except Exception as e:
    print(f"Error: {e}")
    db.rollback()

finally:
    cursor.close()
    db.close()
