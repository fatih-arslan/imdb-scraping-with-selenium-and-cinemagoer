import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup


class WebHandler:
    def __init__(self, url: str = ""):
        self.url = url
        # Provide the path to the chromedriver executable
        self.chromedriver_path = r"C:\Users\Fatih\Desktop\chromedriver.exe"
        self.options = Options()
        self.options.add_experimental_option("detach", True)
        # Initialize Chrome WebDriver
        self.driver = webdriver.Chrome(options=self.options)

    def __click_show_more(self, num_clicks=3):
        # clicks show more button 3 times to get 1000 movies displayed on the page

        self.driver.get(self.url)
        self.driver.maximize_window()

        # Handle cookie decline button if present
        self.__handle_cookie_decline()

        button_locator = (By.XPATH, '/html/body/div[2]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/div[2]/div/span/button')
        wait = WebDriverWait(self.driver, 15)

        successful_clicks = 0
        while successful_clicks < num_clicks:
            try:
                # Scroll to the element
                button = wait.until(EC.presence_of_element_located(button_locator))
                self.driver.execute_script("arguments[0].scrollIntoView({ behavior: 'auto', block: 'center', inline: 'nearest' });", button)

                # Attempt to click the button
                button.click()

                # If the click succeeds, increment the successful clicks counter
                successful_clicks += 1
                print(f"Button clicked {successful_clicks} times.")
            except:
                print("Button not found within the timeout. Retrying...")
                time.sleep(2)

        print(f"Button successfully clicked {num_clicks} times.")

        time.sleep(5)
        page_source = self.driver.page_source
        return page_source

    def get_movie_links(self):
        try:
            movie_links = []
            page_source = self.__click_show_more()
            soup = BeautifulSoup(page_source, 'html.parser')
            # Find the movie list section
            movie_list_section = soup.find('div', class_='ipc-page-grid__item ipc-page-grid__item--span-2')

            # Find all movies
            movies = movie_list_section.find_all('li', class_='ipc-metadata-list-summary-item')
            for movie in movies:
                div = movie.find('div', {'class': 'ipc-metadata-list-summary-item__c'})
                div_2 = div.find('div', {'class': 'ipc-metadata-list-summary-item__tc'})
                div_3 = div_2.find('div', {'class': 'sc-53c98e73-4 gOfInm dli-parent'})
                div_4 = div_3.find('div', {'class': 'sc-53c98e73-3 kttqmq'})
                div_5 = div_4.find('div', {'class': 'sc-43986a27-0 gUQEVh'})
                div_6 = div_5.find('div', {'class': 'ipc-title ipc-title--base ipc-title--title ipc-title-link-no-icon ipc-title--on-textPrimary sc-43986a27-9 gaoUku dli-title'})
                link = div_6.find('a').get('href')
                movie_links.append(link)
            return movie_links
        except:
            return []

    def get_movie_links_with_url(self, url: str):
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


    def __handle_cookie_decline(self):
        # clicks the decline cookies button

        cookie_decline_button_locator = (By.XPATH, '/html/body/div[2]/div/div/div[2]/div/button[1]')
        try:
            cookie_decline_button = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(cookie_decline_button_locator))
            cookie_decline_button.click()
            print("Cookie decline button clicked")
        except TimeoutException:
            print("Cookie decline button not found within the timeout. Continuing without clicking.")
        except NoSuchElementException:
            print("Cookie decline button not found on the page. Continuing without clicking.")

    def close_browser(self):
        self.driver.quit()
