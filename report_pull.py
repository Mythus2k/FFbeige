import requests
from bs4 import BeautifulSoup
import datetime as dt

def get_report(month=str(), year=str()):
    """
    Pulls beige report for given month and year
        month : str
            must be number form (01 or 1)
        
        year : str
            must be the full year (2020)
    """
    if len(month) == 1:
        month = '0'+month

    url = 'https://www.federalreserve.gov/monetarypolicy/beigebook'+year+month+'.htm'
        
    with requests.get(url) as response: 
        soup = BeautifulSoup(response.content, "html.parser")

    if str(soup.h2) == '<h2>Page not found</h2>': 
        return f"No report available for {month}/{year}"
    
    return soup

def save_report(report=BeautifulSoup()):
    """
    Saves report to local folder titled reports
    """
    h2 = str(report.h2).splitlines()[1].split('-')[1].split(' ')
    date = dt.datetime.strptime(h2[-1]+' '+h2[1],'%Y %B')
    month = str(date.month)
    if len(month) == 1: month = '0'+month
    year = str(date.year)

    with open('./reports/BR'+month+year+'.html','w', encoding='utf-8') as br: 
        br.write(report.prettify()) 

def pull_report(filename=str()):
    return BeautifulSoup(open('./reports/'+filename+'.html','r'), "html.parser")

br = pull_report('BR042023')

sections = br.find_all('p')

lines = []
for p in sections:
    lines.append(' '.join(p.text.replace('\n','').split()))

print(lines[0:20])