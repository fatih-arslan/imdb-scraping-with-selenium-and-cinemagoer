from web_handler import WebHandler
from movie_data_service import MovieDataService
import pandas as pd

genres = []

countries = []

languages = []

directors = []

content_rates = []

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

def extract_and_write_data(links):
    global verbal_df
    global numerical_df
    for link in links:
        verbal_data = []
        numerical_data = []
        movie_url = base_url + link
        movie_data_service = MovieDataService(movie_url)

        title = movie_data_service.get_title()
        if title is None:
            continue
        else:
            verbal_data.append(title)
            word_count = len(title.split())
            numerical_data.append(word_count)

        year = movie_data_service.get_release_year()
        if year is None:
            continue
        else:
            verbal_data.append(year)
            numerical_data.append(year)

        content_rate = movie_data_service.get_content_rate()
        if content_rate is None:
            continue
        else:
            verbal_data.append(content_rate)
            numerical_data.append(content_rate)

        duration = movie_data_service.get_duration()
        if duration is None:
            continue
        else:
            verbal_data.append(duration)
            numerical_data.append(duration)

        genre = movie_data_service.get_genre()
        if genre is None:
            continue
        else:
            verbal_data.append(genre)
            numerical_data.append(genre)

        director = movie_data_service.get_director()
        if director is None:
            continue
        else:
            verbal_data.append(director)
            numerical_data.append(director)

        writer = movie_data_service.get_writer()
        if writer is None:
            continue
        else:
            verbal_data.append(writer)
            numerical_data.append(writer)

        producer = movie_data_service.get_producer()
        if producer is None:
            continue
        else:
            verbal_data.append(producer)
            numerical_data.append(producer)

        rating = movie_data_service.get_rating()
        if rating is None:
            continue
        else:
            verbal_data.append(rating)
            numerical_data.append(rating)

        country = movie_data_service.get_country()
        if country is None:
            continue
        else:
            verbal_data.append(country)
            numerical_data.append(country)

        language = movie_data_service.get_language()
        if language is None:
            continue
        else:
            verbal_data.append(language)
            numerical_data.append(language)

        budget = movie_data_service.get_budget()
        if budget is None:
            continue
        else:
            verbal_data.append(budget)
            numerical_data.append(budget)

        gross = movie_data_service.get_total_gross()
        if gross is None:
            continue
        else:
            verbal_data.append(gross)
            numerical_data.append(gross)
        numerical_df.loc[len(numerical_df)] = numerical_data
        verbal_df.loc[len(verbal_df)] = verbal_data






# web_handler = WebHandler(urls[0])
# links_1 = web_handler.get_movie_links()
# print(len(links_1))
# top_1 = links_1[:250]
# print(len(top_1))
# top_2 = links_1[750:]
# print(len(top_2))
# web_handler.close_browser()
# avg_1 = web_handler.get_movie_links_with_url(urls[1])
# print(len(avg_1))
# avg_2 = web_handler.get_movie_links_with_url(urls[2])
# print(len(avg_2))
# avg_3 = web_handler.get_movie_links_with_url(urls[3])
# print(len(avg_3))
# avg_4 = web_handler.get_movie_links_with_url(urls[4])
# print(len(avg_4))
# avg_5 = web_handler.get_movie_links_with_url(urls[5])
# print(len(avg_5))
# web_handler_2 = WebHandler(urls[6])
# links_2 = web_handler_2.get_movie_links()
# bottom_1 = links_2[:250]
# print(len(bottom_1))
# bottom_2 = links_2[750:]
# print(len(bottom_2))
# web_handler_2.close_browser()







