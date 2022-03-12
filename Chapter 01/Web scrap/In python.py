

import requests
from bs4 import BeautifulSoup

URL = "https://forecast.weather.gov/MapClick.php?x=276&y=148&site=lox&zmx=&zmy=&map_x=276&map_y=148"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")