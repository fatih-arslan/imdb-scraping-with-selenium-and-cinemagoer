from web_handler import WebHandler
from movie_data_service import MovieDataService
import pandas as pd
from sklearn.preprocessing import LabelEncoder


movie_features = ['title', 'release year', 'content rating', 'duration', 'genre', 'director',
                  'writer', 'producer', 'rating', 'country', 'language', 'budget', 'total gross']

urls = [
    "https://www.imdb.com/search/title/?sort=user_rating,desc&groups=top_1000&count=250",
    "https://www.imdb.com/list/ls051720433/?sort=list_order,asc&st_dt=&mode=detail&page=1",
    "https://www.imdb.com/list/ls051720433/?st_dt=&mode=detail&page=2&sort=list_order,asc",
    "https://www.imdb.com/list/ls051720433/?st_dt=&mode=detail&page=3&sort=list_order,asc",
    "https://www.imdb.com/list/ls051720433/?st_dt=&mode=detail&page=4&sort=list_order,asc",
    "https://www.imdb.com/list/ls051720433/?st_dt=&mode=detail&page=5&sort=list_order,asc",
    "https://www.imdb.com/search/title/?sort=user_rating,desc&groups=bottom_1000&count=250",
]

verbal_df = pd.DataFrame(columns=movie_features)
numerical_df = pd.DataFrame(columns=movie_features)

base_url = "https://www.imdb.com/"
count = 0
def extract_and_write_data(links: list):
    global verbal_df
    global numerical_df
    global count
    for link in links:
        try:
            movie_url = base_url + link
            movie_data_service = MovieDataService(movie_url)
            verbal_data = []
            numerical_data = []

            title = movie_data_service.get_title()
            if title is None:
                print("title none")
                continue
            else:
                verbal_data.append(title)
                word_count = len(title.split())
                numerical_data.append(word_count)

            year = movie_data_service.get_release_year()
            if year is None:
                print("year none")
                continue
            else:
                verbal_data.append(year)
                numerical_data.append(year)

            content_rate = movie_data_service.get_content_rate()
            if content_rate is None:
                print("content rate none")
                continue
            else:
                verbal_data.append(content_rate)
                numerical_data.append(content_rate)

            duration = movie_data_service.get_duration()
            if duration is None:
                print("duration none")
                continue
            else:
                verbal_data.append(duration)
                numerical_data.append(duration)

            genre = movie_data_service.get_genre()
            if genre is None:
                print("genre none")
                continue
            else:
                verbal_data.append(genre)
                numerical_data.append(genre)

            director = movie_data_service.get_director()
            if director is None:
                print("director none")
                continue
            else:
                verbal_data.append(director)
                numerical_data.append(director)

            writer = movie_data_service.get_writer()
            if writer is None:
                print("writer none")
                continue
            else:
                verbal_data.append(writer)
                numerical_data.append(writer)

            producer = movie_data_service.get_producer()
            if producer is None:
                print("producer none")
                continue
            else:
                verbal_data.append(producer)
                numerical_data.append(producer)

            rating = movie_data_service.get_rating()
            if rating is None:
                print("rating none")
                continue
            else:
                verbal_data.append(rating)
                numerical_data.append(rating)

            country = movie_data_service.get_country()
            if country is None:
                print("country none")
                continue
            else:
                verbal_data.append(country)
                numerical_data.append(country)

            language = movie_data_service.get_language()
            if language is None:
                print("language none")
                continue
            else:
                verbal_data.append(language)
                numerical_data.append(language)

            budget = movie_data_service.get_budget()
            if budget is None:
                print("budget none")
                continue
            else:
                verbal_data.append(budget)
                numerical_data.append(budget)

            gross = movie_data_service.get_total_gross()
            if gross is None:
                print("gross none")
                continue
            else:
                verbal_data.append(gross)
                numerical_data.append(gross)
            count += 1
            print(f"{count}")
            numerical_df.loc[len(numerical_df)] = numerical_data
            verbal_df.loc[len(verbal_df)] = verbal_data
        except:
            continue


# Label encoding for 'content rating'

# numerical_df['content rating'] = label_encoder.fit_transform(numerical_df['content rating'])

# numerical_df = pd.get_dummies(numerical_df, columns=['genre', 'director', 'country', 'writer', 'producer', 'language'],
#                               prefix=['genre', 'director', 'country', 'writer', 'producer', 'language'])

web_handler = WebHandler(urls[0])
links_1 = web_handler.get_movie_links()
while len(links_1) == 0:
    links_1 = web_handler.get_movie_links()
print(len(links_1))
top_1 = links_1[:250]
top_2 = links_1[750:]
web_handler.close_browser()
avg_1 = MovieDataService.get_movie_links_with_url(urls[1])
avg_2 = MovieDataService.get_movie_links_with_url(urls[2])
avg_3 = MovieDataService.get_movie_links_with_url(urls[3])
avg_4 = MovieDataService.get_movie_links_with_url(urls[4])
avg_5 = MovieDataService.get_movie_links_with_url(urls[5])
web_handler_2 = WebHandler(urls[6])
links_2 = web_handler_2.get_movie_links()
while len(links_2) == 0:
    links_2 = web_handler.get_movie_links()
print(len(links_2))
bottom_1 = links_2[:250]
bottom_2 = links_2[len(links_2)-250:]
web_handler_2.close_browser()
print(len(top_1))
print(len(top_2))
print(len(avg_1))
print(len(avg_2))
print(len(avg_3))
print(len(avg_4))
print(len(avg_5))
print(len(bottom_1))
print(len(bottom_2))
all_movies = top_1 + top_2 + avg_1 + avg_2 + avg_3 + avg_4 + avg_5 + bottom_1 + bottom_2
print(f"all movies: {len(all_movies)}")
extract_and_write_data(all_movies)

excel_file_path = 'verbal_data.xlsx'
excel_file_path_2 = 'numerical_data.xlsx'

label_encoder = LabelEncoder()

features_to_encode = ['title', 'content rating', 'genre', 'director', 'writer', 'producer', 'country', 'language']

for feature in features_to_encode:
    # Check if the feature exists in the DataFrame and has the 'object' dtype
    if feature in numerical_df.columns:
        numerical_df[feature] = label_encoder.fit_transform(numerical_df[feature])

# Write the DataFrame to an Excel file
verbal_df.to_excel(excel_file_path, index=False)
numerical_df.to_excel(excel_file_path_2, index=False)



