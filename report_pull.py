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

    if int(year)>2016: url = 'https://www.federalreserve.gov/monetarypolicy/beigebook'+year+month+'.htm'
    elif int(year)>=2011: url = 'https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook'+year+month+'.htm'
    elif int(year)<2011: 
        print(f"Reports prior to 2010 are not prepared yet")
        return None
    else: return f"unkown error getting report from Federal Reserve"
        
    soup = BeautifulSoup(requests.get(url).content, "html.parser")

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

    with open('./reports/BR'+month+year+'.html','w', encoding='utf-8') as file: 
        file.write(report.prettify()) 

def pull_local_report(month=str(),year=str()):
    """
    Pulls beige report for given month and year
        month : str
            must be number form (01 or 1)
        
        year : str
            must be the full year (2020)
    """
    if len(month) == 1: month = '0'+month
    return BeautifulSoup(open('./reports/BR'+year+month+'.html','r'), "html.parser")

def version_check(report=BeautifulSoup()):
    version_three = 2017
    version_two = 2011
    version_one = 1996
    report_year = int(report.title.get_text().strip()[-4:])
    
    if report_year >= version_three: return 3
    if report_year >= version_two: return 2
    if report_year >= version_one: return 1
    return 0 

def clean_report(report=BeautifulSoup()):
    """
    Cleans up html and gives the report broken up into sections
    """
    version = version_check(report)

    if version == 1:
        print('not ready to clean the report version 1 yet')
        return None
    
    if version == 2:
        print('not ready to clean the report version 2 yet')
        return None
    
    if version == 3:
        return vThreeClean(report)
    
def vThreeClean(report=BeautifulSoup()):
    
    Bank = 'Overall Market'
    for line in report.find(id='article').find_all('p')[13:16]:
        
        if line.strong == None:
            Bank = str(line.a).split('>')[0].split('=')[-1].replace('"','').capitalize()
            Bank = 'Federal Reserve Bank of ' + Bank

        else:
            print(line.get_text())



br = pull_local_report('04','2023')
clean_report(br)