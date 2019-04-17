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

def get_links(query):
    '''
    Getting 10 first links from Google and
    link to Google images search
    '''
    def parse_url(url):
        '''
        Replace query symbols for double-encoded strings
        '''
        replaces = { '%3F':'?','%3D':'=','%2520':'%20','&amp;':"&"}
        for key, value in replaces.items():
            url = url.replace(key,value)
        return url
        
    url = 'https://google.com/search?q=' + quote(query)
    response = requests.get(url)
    res = response.text
    links = [ each.split('&amp')[0] for each in res.split('/url?q=')[1:] ]
    links = [parse_url(link) for link in links if link.count('webcache.googleusercontent.com')==0]
    images_link = res.split("/search?q=")[1:]
    images_link = [ each.split('">Images')[0] for each in images_link]
    im_l = parse_url(images_link[np.argmin([len(each) for each in images_link])])
    im_l = 'https://google.com/search?q=' + im_l
            
    return list(set(links)), im_l

def get_images(link,num=10):
    '''
    Searching Google for images and getting
    URLs of first 'num' images
    '''
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
    '''
    Running k_means on each image RGB data,
    assembling centroids into one palette,
    running k_means on that palette again
    and dropping colors from clusters that 
    have elements in less than 5 images out 
    of total. Returning filtered patette 
    sorted by luminousity (approximately) 
    '''
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
    '''
    Wrapper function to get BS4 object from link
    '''
    response = requests.get(link.replace("https","http"), verify=False,timeout=1)
    if response.status_code==200:
        global soups
        soups[num] = BeautifulSoup(response.text, 'html.parser')

def cook_soup(s):
    '''
    Obtaining most reasonable text content from
    BS4 object. First try everything from P or
    BLOCKQUOTE elements. If page is not made that
    way then filter out all possibly noizy elements
    and get what's left in the BODY
    '''
    text = ""

    if not isinstance(s, BeautifulSoup):
        return ""

    for each in s.find_all(['script','style','h1','h2','h3','h4','textarea','input','button','form']):
        each.decompose()

    for each in s.find_all(['p','blockquote']):
        text += each.get_text()
        
    if text !="":
        return text

    for body in s.find_all('body'):
        for each in body.contents:
            try:
                text += ' ' + each.get_text()
            except:
                pass
    return text

def clean_text(t):
    '''
    Remove newlines and punctuation symbols
    keeping dash and $
    '''
    result = t.replace("\n","").lower()
    result = "".join([ char if char not in '!"%&\'()*,./:;<=>?[\\]^_`{|}~' else ' ' for char in result ])
    return result

def get_words(query,id,clarify):
    '''
    Process function to get color palette
    and bag of words from Google based on
    query and clarifying keywords
    '''

    from run import log_progress
    
    links, images_link = get_links(query+' '+clarify)
    
    global soups
    soups = []
    threads = []

    log_progress(id,1)
    
    for ind,each in enumerate(links):
        soups.append("")
        threads.append(threading.Thread(target=get_soup, args=([each,ind])))

    for num in range(0,len(threads)):
        threads[num].start()
                   
    for num in range(0,len(threads)):
        threads[num].join()

    contents = []

    log_progress(id,2)

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

    log_progress(id,3)

    imgs = get_images(images_link)

    cmap = get_cumulative_palette(imgs)

    return words_in_order, cmap

    


