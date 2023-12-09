from web_handler import WebHandler
from movie_data_service import MovieDataService
from imdb import Cinemagoer



urls = [
    "https://www.imdb.com/search/title/?sort=user_rating,desc&groups=top_1000&count=250",
    "https://www.imdb.com/search/title/?sort=release_date,desc&groups=bottom_1000&count=250",
    "https://www.imdb.com/list/ls051720433/"
]

base_url = "https://www.imdb.com/"

web_handler = WebHandler(urls[0])
links = web_handler.get_movie_links()[:10]
print(len(links))
for link in links:
    movie_url = base_url + link
    movie_data_service = MovieDataService(movie_url)
    title = movie_data_service.get_title()
    year = movie_data_service.get_release_year()
    content_rate = movie_data_service.get_content_rate()
    duration = movie_data_service.get_duration()
    director = movie_data_service.get_director()
    writer = movie_data_service.get_writer()
    producer = movie_data_service.get_producer()
    genre = movie_data_service.get_genre()
    rating = movie_data_service.get_rating()
    country = movie_data_service.get_country()
    language = movie_data_service.get_language()
    budget = movie_data_service.get_budget()
    gross = movie_data_service.get_total_gross()
    print(f"title: {title}, year: {year}, content rate: {content_rate}, duration: {duration}, director: {director}, "
          f"writer: {writer}, producer: {producer}, genre: {genre}, rating: {rating}, country: {country}, language: {language}, "
          f"budget: {budget}, gross: {gross}")





