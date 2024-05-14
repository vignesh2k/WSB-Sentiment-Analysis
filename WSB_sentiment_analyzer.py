from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from psaw import PushshiftAPI
import datetime as dt
import re

api = PushshiftAPI()

regex = re.compile('[^a-zA-Z^$]')
disallowed_char = ['$','M','K','B','T']  # removes denominations that slip through

def validate_date(date_text):
    try:
        dt.datetime.strptime(date_text, '%d %m %Y')
        return True
    except ValueError:
        return False

def get_submissions(subreddit, start_time, end_time):
    try:
        return list(api.search_submissions(after=start_time,
                                           before=end_time,
                                           subreddit=subreddit,
                                           filter=['url', 'author', 'title', 'subreddit']))
    except Exception as e:
        print(f"Error fetching submissions: {e}")
        return []

def parse_submission_titles(submissions):
    stock_dict = {}
    for submission in submissions:
        words = submission.title.upper().split()
        words = [regex.sub('', word) for word in words]
        words = [word for word in words if word and word[0] == '$' and word[1] not in disallowed_char]

        stock_tickers = list(set(filter(lambda word: word.lower().startswith('$'), words)))
        for stock in stock_tickers:
            if not any(char.isdigit() for char in stock):
                stock = stock[1:]
                stock_dict[stock] = stock_dict.get(stock, 0) + 1
    return stock_dict

def wsbscraper():
    subreddit_input = 'wallstreetbets'
    
    usrdate = input('What date do you wish to analyze? (DD MM YYYY): ')
    while not validate_date(usrdate):
        usrdate = input('Invalid date format. Please enter the date in DD MM YYYY format: ')
    
    nums = usrdate.split()
    init_time = int(dt.datetime(int(nums[2]), int(nums[1]), int(nums[0])).timestamp())
    end_time = int((dt.datetime(int(nums[2]), int(nums[1]), int(nums[0])) + dt.timedelta(days=1)).timestamp())
    
    submissions = get_submissions(subreddit_input, init_time, end_time)
    stock_dict = parse_submission_titles(submissions)
    
    sorted_stock_dict = dict(reversed(sorted(stock_dict.items(), key=lambda item: item[1])))
    tickers = list(sorted_stock_dict.keys())[:3]
    print(tickers)
    return tickers

def fetch_news(tickers):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    for ticker in tickers:
        url = finviz_url + ticker
        try:
            req = Request(url=url, headers={'user-agent': 'news-scraper-app'})
            response = urlopen(req)
            html = BeautifulSoup(response, features='html.parser')
            news_table = html.find(id='news-table')
            news_tables[ticker] = news_table
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
    return news_tables

def parse_news(news_tables):
    parsed_data = []
    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            title = row.a.text
            date_data = row.td.text.split(' ')
            if len(date_data) == 1:
                time = date_data[0]
                date = None
            else:
                date = date_data[0]
                time = date_data[1]
            parsed_data.append([ticker, date, time, title])
    return pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

def sentimentanalyses(tickers):
    news_tables = fetch_news(tickers)
    df = parse_news(news_tables)
    
    vader = SentimentIntensityAnalyzer()
    df['compound'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])
    df['date'] = pd.to_datetime(df.date).dt.date
    
    plt.figure(figsize=(10,8))
    mean_df = df.groupby(['ticker', 'date']).mean()
    mean_df = mean_df.unstack()
    mean_df = mean_df.xs('compound', axis='columns').transpose()
    mean_df.plot(kind='bar')
    plt.show()

if __name__ == "__main__":
    tickers = wsbscraper()
    if tickers:
        sentimentanalyses(tickers)
