import re
import requests
from bs4 import BeautifulSoup
from imdb import Cinemagoer


class MovieDataService:
    def __init__(self, url: str):
        self.url = url
        self.ia = Cinemagoer()
        self.movie_id = self.__extract_imdb_id(self.url)
        self.movie = self.ia.get_movie(self.movie_id)

    @staticmethod
    def get_movie_links_with_url(url: str):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        div = soup.find('div', {'class': 'lister list detail sub-list'})
        div_2 = div.find('div', {'class': 'lister-list'})
        movie_divs = div_2.find_all('div', {'class': 'lister-item mode-detail'})
        movie_links = []
        for movie in movie_divs:
            content = movie.find('div', {'class': 'lister-item-content'})
            header = content.find('h3', {'class': 'lister-item-header'})
            link = header.find('a').get('href')
            movie_links.append(link)
        return movie_links

    @staticmethod
    def get_soup(url: str):
        headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, 'html.parser')
        return soup

    def get_title(self):
        try:
            return self.movie['original title']
        except:
            return None

    def get_release_year(self):
        try:
            return self.movie['year']
        except:
            return None

    def get_duration(self):
        try:
            return self.movie['runtimes'][0]
        except:
            return None

    def get_content_rate(self):
        try:
            soup = self.get_soup(self.url)
            header_data_div = soup.find('div', {'class': 'sc-69e49b85-0 jqlHBQ'})
            header_data_list = header_data_div.find('ul', {'class': 'ipc-inline-list ipc-inline-list--show-dividers sc-d8941411-2 cdJsTz baseAlt'})
            header_data = header_data_list.find_all('li', {'class': 'ipc-inline-list__item'})
            try:
                content_rate = header_data[1].text if len(header_data) == 3 else 'Not Rated'
            except:
                content_rate = 'Not Rated'
            return content_rate
        except:
            return 'Not Rated'

    def get_director(self):
        try:
            return self.movie['directors'][0]['name']
        except:
            return None

    def get_writer(self):
        try:
            return self.movie['writers'][0]['name']
        except:
            return None

    def get_producer(self):
        try:
            return self.movie['producer'][0]['name']
        except:
            return 'No producer'


    def get_genre(self):
        try:
            return self.movie['genres'][0]
        except:
            return None

    def get_rating(self):
        try:
            return self.movie['rating']
        except:
            return None

    def get_country(self):
        try:
            return self.movie['countries'][0]
        except:
            return None

    def get_language(self):
        try:
            return self.movie['languages'][0]
        except:
            return None

    def get_budget(self):
        try:
            budget = self.movie['box office']['Budget']
            try:
                numeric_budget = self.__extract_numeric_value(budget)
                return numeric_budget
            except:
                return budget
        except:
            return None


    def get_total_gross(self):
        try:
            soup = self.get_soup(self.url)
            gross_data_section = soup.find('section', {'cel_widget_id': 'StaticFeature_BoxOffice'})
            gross_data_div = gross_data_section.find('div', {'class': 'sc-f65f65be-0 bBlII'})
            data_list = gross_data_div.find('ul', {
                'class': 'ipc-metadata-list ipc-metadata-list--dividers-none ipc-metadata-list--compact sc-ae184607-0 cfyqRt ipc-metadata-list--base'})
            data_items = data_list.find_all('li')
            gross_value = data_items[-1].text
            gross_value = self.__extract_numeric_value(gross_value)
            return gross_value
        except:
            return None

    def __extract_numeric_value(self, currency_string: str):
        # Define a regular expression pattern to match the numeric part
        pattern = r'[\D]*([\d,]+)[\D]*'

        # Use re.search to find the pattern in the string
        match = re.search(pattern, currency_string)

        # If a match is found, return the numeric part as a string
        if match:
            numeric_part = match.group(1)
            return numeric_part
        else:
            return None

    def __extract_imdb_id(self, url: str):
        # Define a regular expression pattern to match the IMDb numeric ID
        pattern = r'/title/tt(\d+)/'

        # Use re.search to find the pattern in the URL
        match = re.search(pattern, url)

        # If a match is found, return the numeric part of the IMDb ID
        if match:
            numeric_id = match.group(1)
            return numeric_id
        else:
            return None

    def __get_minute_value(self, time_string: str):
        # gets the minute value from a string like "2h22m"
        pattern = re.compile(r'(\d+)h\s*(\d+)m')

        match = pattern.search(time_string)

        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))

            total_minutes = hours * 60 + minutes
            return total_minutes
        else:
            return None

