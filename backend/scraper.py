# backend/scraper.py

import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

MIN_TEXT_LEN = 200  # below this = likely dynamic content


def fetch_requests(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            return resp.text
    except:
        pass
    return ""


def fetch_selenium(url: str) -> str:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    driver.get(url)

    # Scroll to bottom to force content load
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(6):  # scroll a few times
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    html = driver.page_source
    driver.quit()
    return html


def fetch_page(url: str) -> str:
    html = fetch_requests(url)
    if len(html) < MIN_TEXT_LEN:
        return fetch_selenium(url)
    return html


def parse_html(html: str):
    return BeautifulSoup(html, "html.parser")
