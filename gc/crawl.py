import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import Counter
import threading 

soups = []

def get_words(query,id,clarify):
    def get_links(query,where='google'):
        def parse_url(url):
            replaces = { '%3F':'?','%3D':'=','%2520':'%20'}
            for key, value in replaces.items():
                url = url.replace(key,value)
            return url
        
        engines = {"google":'https://google.com/search?q=','pinterest':"","yandex":""}
        url = engines[where] + query
        response = requests.get(url)
        res = response.text
        if where == 'google':
            links = [ each.split('&amp')[0] for each in res.split('/url?q=')[1:] ]
            links = [parse_url(link) for link in links if link.count('webcache.googleusercontent.com')==0]
            
        return list(set(links))

    def parse_url(url):
        replaces = { '%3F':'?','%3D':'=','%2520':'%20'}
        for key, value in replaces.items():
            url = url.replace(key,value)
        return url
        
    def get_soups(links):
        soups = []
        #responces = []
        for link in links:
            if link[0:4]=='http':
                try:
                    response = requests.get(link.replace('https','http'),verify=False,timeout=5)
                    if response.status_code==200:
                        #responces.append(response.text)
                        soups.append(BeautifulSoup(response.text, 'html.parser'))
                except:
                    pass
        return soups #, responces
    
    def get_soup(link,num):
        response = requests.get(link.replace("https","http"), verify=False,timeout=5)
        if response.status_code==200:
            global soups
            soups[num] = BeautifulSoup(response.text, 'html.parser')

    def cook_soup(s):
        text = ""

        if s==None:
            return ""

        for each in s.find_all(['p','blockquote']):
            text += each.get_text()
        if text !="":
            #print('found P and QUOTE')
            return text
        
        for each in s.find_all(['script','style','h1','h2','h3','h4']):
            each.decompose()
        text = s.find('body').get_text()
        if text!="":
            #print('got content from body')
            return text

        return ""

    def clean_text(t):
        result = t.replace("\n","").lower()
        result = "".join([ char if char not in '0123456789/\:,.;-!?&()#"%[»]—' else ' ' for char in result ])
        return result

    from run import log_progress
    
    links = get_links(query+' '+clarify)
    
    global soups
    soups = []
    threads = []

    log_progress(id)
    
    for ind,each in enumerate(links):
        soups.append(None)
        threads.append(threading.Thread(target=get_soup, args=([each,ind])))

    for num in range(0,len(threads)):
        threads[num].start()
                   
    for num in range(0,len(threads)):
        threads[num].join()

    #soups = get_soups(links)

    contents = []

    log_progress(id)

    for each in soups:
        contents.append(clean_text(cook_soup(each)))

    total_count = Counter([
        word for word in " ".join(contents).split(" ")
        if word!='' and word not in set(stopwords.words('english'))])

    words_in_order = [ 
        each[0] for each in total_count.most_common(len(total_count)) 
        if each[0] !=query.lower()]

    return words_in_order

    


