import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import Counter
import threading 

from PIL import Image
from sklearn.cluster import k_means

from urllib.parse import quote

soups = []

def get_words(query,id,clarify):
    def get_links(query,where='google'):
        def parse_url(url):
            replaces = { '%3F':'?','%3D':'=','%2520':'%20','&amp;':"&"}
            for key, value in replaces.items():
                url = url.replace(key,value)
            return url
        
        engines = {"google":'https://google.com/search?q=','pinterest':"","yandex":""}
        url = engines[where] + quote(query)
        response = requests.get(url)
        res = response.text
        if where == 'google':
            links = [ each.split('&amp')[0] for each in res.split('/url?q=')[1:] ]
            links = [parse_url(link) for link in links if link.count('webcache.googleusercontent.com')==0]
            images_link = res.split("/search?q=")[1:]
            images_link = [ each.split('">Images')[0] for each in images_link]
            im_l = parse_url(images_link[np.argmin([len(each) for each in images_link])])
            im_l = 'https://google.com/search?q=' + im_l
            
        return list(set(links)), im_l

    def get_images(link,num=10):
        res = requests.get(link)
        im_res = res.text
        im_soup = BeautifulSoup(im_res, 'html.parser')
        srcs = [ each['src'] for each in im_soup.find_all('img')]
        imgs = []
        for each in srcs:
            r = requests.get(each, stream=True)
            if r.status_code==200:
                imgs.append(Image.open(r.raw))
        return imgs[0:num]

    def get_cumulative_palette(imgs):
        big_pal = []
        bins = [0]
        for image in imgs:
            tmp = image.convert('P', palette = Image.PERSPECTIVE, colors=256)
            tmp = np.array(tmp.convert('RGB'))
            c = k_means(tmp.reshape(tmp.shape[0]*tmp.shape[1],3),16,init='random')
            palette = np.round(c[0]).astype(int)
            big_pal.append(palette)
            bins.append(bins[-1]+c[0].shape[0])
    
        big_palette = np.vstack(big_pal)

        cols = 16

        c = k_means(big_palette,cols)

        span = []
        
        for each in np.arange(0,cols):
            span.append(np.unique(np.digitize(np.argwhere(c[1]==each),bins)).size)

        centr_to_drop = np.arange(0,cols)[np.array(span)<5]
        
        filtered_palette = np.array([
            each for ind,each in enumerate(big_palette) 
            if c[1][ind] not in centr_to_drop ])

        f_indxs = np.argsort(np.sum(filtered_palette,axis=1))
    
        return filtered_palette[f_indxs]
        
    def get_soup(link,num):
        response = requests.get(link.replace("https","http"), verify=False,timeout=1)
        if response.status_code==200:
            global soups
            soups[num] = BeautifulSoup(response.text, 'html.parser')

    def cook_soup(s):
        text = ""

        if not isinstance(s, BeautifulSoup):
            return ""

        for each in s.find_all(['p','blockquote']):
            text += each.get_text()
        if text !="":
            return text
        
        for each in s.find_all(['script','style','h1','h2','h3','h4']):
            each.decompose()
        for each in s.find_all('body'):
            text += each.get_text()
        
        return text

    def clean_text(t):
        result = t.replace("\n","").lower()
        result = "".join([ char if char not in '!"%&\'()*,./:;<=>?[\\]^_`{|}~' else ' ' for char in result ])
        return result

    from run import log_progress
    
    links, images_link = get_links(query+' '+clarify)
    
    global soups
    soups = []
    threads = []

    log_progress(id)
    
    for ind,each in enumerate(links):
        soups.append("")
        threads.append(threading.Thread(target=get_soup, args=([each,ind])))

    for num in range(0,len(threads)):
        threads[num].start()
                   
    for num in range(0,len(threads)):
        threads[num].join()

    contents = []

    log_progress(id)

    for each in soups:
        contents.append(clean_text(cook_soup(each)))

    sw = set(stopwords.words('english'))
    sw.add('also')
    sw.add('loading')

    total_count = Counter([
        word for word in (" ".join(contents)).split(" ")
        if len(word)>1 and word not in sw])

    words_in_order = [ 
        each[0] for each in total_count.most_common(len(total_count)) 
        if each[0] !=query.lower()]

    log_progress(id)

    imgs = get_images(images_link)

    cmap = get_cumulative_palette(imgs)

    return words_in_order, cmap

    


