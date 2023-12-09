import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class WebHandler:
    def __int__(self, url: str):
        self.url = url
        # Provide the path to the chromedriver executable
        self.chromedriver_path = r"C:\Users\Fatih\Desktop\chromedriver.exe"
        self.options = Options()
        self.options.add_experimental_option("detach", True)
        # Initialize Chrome WebDriver
        self.driver = webdriver.Chrome(options=self.options)

    def click_show_more(self, num_clicks=3):
        # clicks show more button 3 times to get 1000 movies displayed on the page

        self.driver.get(self.url)
        self.driver.maximize_window()

        # Handle cookie decline button if present
        self.handle_cookie_decline()

        button_locator = (By.XPATH, 'your_XPATH_for_show_more_button')
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
            except TimeoutException:
                print("Button not found within the timeout. Retrying...")
                time.sleep(2)

        print(f"Button successfully clicked {num_clicks} times.")

        time.sleep(5)
        page_source = self.driver.page_source
        return page_source

    def handle_cookie_decline(self):
        # clicks the decline cookies button

        cookie_decline_button_locator = (By.XPATH, 'your_XPATH_for_cookie_decline_button')
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

