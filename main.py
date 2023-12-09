from web_handler import WebHandler
from movie_data_service import MovieDataService
from imdb import Cinemagoer

urls = [
    "https://www.imdb.com/title/tt0108052/?ref_=sr_t_4",
    "https://www.imdb.com/title/tt0111161/?ref_=sr_t_1",
    "https://www.imdb.com/title/tt9179430/?ref_=sr_t_115",
    "https://www.imdb.com/title/tt0119698/?ref_=sr_t_113",
    "https://www.imdb.com/title/tt0019254/?ref_=sr_t_300",
    "https://www.imdb.com/title/tt0947798/?ref_=sr_t_354",
    "https://www.imdb.com/title/tt0095953/?ref_=sr_t_375",
    "https://www.imdb.com/title/tt0365748/?ref_=sr_t_500",
    "https://www.imdb.com/title/tt0087553/?ref_=sr_t_700",
    "https://www.imdb.com/title/tt0230011/?ref_=ttls_li_tt"
]


for url in urls:
    movie_data_service = MovieDataService(url=url)
    title = movie_data_service.get_title()
    release_year = movie_data_service.get_release_year()
    content_rate = movie_data_service.get_content_rate()
    duration = movie_data_service.get_duration()
    director = movie_data_service.get_director()
    writer = movie_data_service.get_writer()
    producer = movie_data_service.get_producer()
    genre = movie_data_service.get_genre()
    country = movie_data_service.get_country()
    lang = movie_data_service.get_language()
    rating = movie_data_service.get_rating()
    budget = movie_data_service.get_budget()
    total_gross = movie_data_service.get_gross_value()

    print(f"title: {title}, year: {release_year}, content_rate: {content_rate}, duration = {duration}, director: {director}, writer: {writer},  producer: {producer} genre: {genre}, country: {country}, language: {lang}, rating: {rating} budget: {budget}, gross: {total_gross}")


# ia = Cinemagoer()
#
# movie = ia.get_movie('0019254')
# for key in movie.keys():
#     print(f"{key}: {movie[key]}")
