from bs4 import BeautifulSoup as bs
from selenium import webdriver
import time
import csv
import pandas as pd

#Create csv
outfile = open("reddit-Cybertruck-sub1.csv", "w", newline='')
writer = csv.writer(outfile)

#Blank df
df = pd.DataFrame(columns=['topic','post_info'])

# I used Edge; you can use whichever browser you like.
browser = webdriver.Edge('C:\Windows\System32\msedgedriver.exe')

# Tell Selenium to get the URL you're interested in.
browser.get("https://www.reddit.com/r/cybertruck/")
for i in range(1, 20):
    # * With this line You can skip infinite scroll
    #scroll to Down
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    #Wait for 4 secend
    time.sleep(4)
# ... code ...
# Now that the page is fully scrolled, grab the source code.
source_data = browser.page_source

# Throw your source into BeautifulSoup and start parsing!
bs_data = bs(source_data, features="html.parser")

# Get the list of post
for item in bs_data.select('.Post'):
    topic = item.select('._eYtD2XCVieq6emjKBH3m')[0].get_text()
    post_info = item.select('.cZPZhMe-UCZ8htPodMyJ5')[0].get_text()
#to dataframe
    df2 = pd.DataFrame([[topic, post_info]],columns=['topic', 'post_info'])
    df = df.append(df2,ignore_index=True)

#save to CSV
df.to_csv('reddit-Cybertruck-sub1.csv')
outfile.close()









