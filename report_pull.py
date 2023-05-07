import requests
from bs4 import BeautifulSoup

"""
########### Report Pull Library ###########
Pull beige reports from the federal reserve
and clean them into a dictionary of each
section.

Sections are Overall Market and then the
federal reserve banks.

Note that the spelling of the Federal Reserve 
banks is not consistent, to access the 
sections for each bank use keys().
 ^ This will be cleaned up in the future

Basic Structure:
 > month = '01'
 > year = '2023'
 > beige_report = get_report(month, year)
 > report = clean_report(beige_report)

 The 'report' variable is a dict() with a dict()
 First layer is the Overall Market and Feds
 Second layer is the relevant sections reported
"""

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
    else: 
        print(f"unkown error getting report from Federal Reserve")
        return None
        
    soup = BeautifulSoup(requests.get(url).content, "html.parser")

    if str(soup.h2) == '<h2>Page not found</h2>': 
        print(f"No report available for {month}/{year}")
        return None
    
    return soup

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
    Cleans up html and gives the report broken up into a dictionary
    """
    version = version_check(report)

    if version == 1:
        print('not ready to clean the report version 1 yet (1996-2010)')
        return vOneClean(report)
    
    if version == 2:
        print('not ready to clean the report version 2 yet (2011-2016)')
        return vTwoClean(report)
    
    if version == 3:
        return vThreeClean(report)
    
def vOneClean(report=BeautifulSoup()):
    return None

def vTwoClean(report=BeautifulSoup()):
    return None

def vThreeClean(report=BeautifulSoup()):    
    bank = 'Overall Market'
    dissected_report = {bank: {}}

    report = report.find(id='article')
    divs = report.find_all('div')
    for div in divs: div.extract()

    for line in report.find_all('p'):
        
        if line.strong == None:
            if len(line.get_text().strip()) > 0: None

            else:
                bank = str(line.a).split('>')[0].split('=')[-1].replace('"','').capitalize()
                bank = 'Federal Reserve Bank of ' + bank
                dissected_report[bank] = {}

        else:
            split_section = line.get_text().splitlines()
            if len(split_section) != 2:
                print(f"Irregularity in report_pull.py, line 123.\n<p> had abnormal structure and was not added: \n{split_section}")

            else:
                header = split_section[0].strip()
                text = split_section[1].strip()

                dissected_report[bank][header] = text

    return dissected_report

if __name__ == '__main__':
    month = '09'
    year = '2017'

    report = get_report(month,year)
    report = clean_report(report)