

import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import pandas as pd

URL = "https://forecast.weather.gov/MapClick.php?x=276&y=148&site=lox&zmx=&zmy=&map_x=276&map_y=148"
page = requests.get(URL)
#print(page)

weather_soup = BeautifulSoup(page.content, "html.parser")


weather_day = weather_soup.find_all("div", class_="col-sm-2 forecast-label")
weather_forecast = weather_soup.find_all("div", class_="col-sm-10 forecast-text")

w = []
for weather1 in weather_forecast:
    w.append(weather1.text)

w2 = []
for weather2 in weather_day:
     w2.append(weather2.text)



start_date = date(2022, 3, 11) 
end_date = date(2022, 3, 17)    

delta = end_date - start_date

date2 = []
for i in range(delta.days + 1):
    date1 = start_date + timedelta(days=i) 
    date2.append(date1)

#print(type(start_date))
#time_difference= timedelta(days= 1 )
#print(start_date + time_difference)
# weather3 = pd.DataFdaysrame({
#         "date": date2,
#          "day": w,
#          "desc": w2
#     })
# print(weather3)
print(len(date2))

# print(delta.days)
# print(start_date + timedelta(days=30))