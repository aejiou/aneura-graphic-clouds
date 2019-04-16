import numpy as np
from itertools import cycle
import pickle

from sklearn.cluster import k_means

def log_progress(id,step,message=None):
    '''
    Writing short message to a temporary file
    for user to see progress
    '''
    stages = [
    'Getting the links from search engine','Getting content from each of the links',
    'Counting words','Getting color palette','Mapping','Rendering the final image']
    f = open("static/tmp/"+id+".log", "w")
    if message==None:
        message = stages[step]
    f.write("Step {} of {}: {}...".format(step+1,len(stages),message))
    f.close()


def generate_image(form):
    '''
    Workflow function for image creation
    '''
    # Helper functions for text transformation
    def upper(w):
        return w.upper()
    def lower(w):
        return w.lower()
    def cap(w):
        return w[:1].upper()+w[1:].lower()
    def rand(w):
        t = [upper,lower,cap][np.random.randint(0,3)]
        return t(w)

    #Helper function to translate DOM true/false to boolean
    t_f = lambda x: True if x=='true' else False

    from src import Canvas, Droplet
    from crawl import get_words

    log_progress(form['identifier'],0)

    w, h = int(form['im_width']), int(form['im_height'])

    '''
    by how much the mask will be smaller than the actual image
    needed for better speed as the best claculation is done on
    the mask
    '''
    if t_f(form['mask']): 
        reducer = 800
    else:
        reducer = 400

    #Define region to look for a place for text at a time
    partial = (0.35,1) if t_f(form['mask']) else (0.5,0.5)

    #colors from dark to light or vice versa
    if t_f(form['inverted']):
        inverter = -1
    else:
        inverter = 1

    #defining fonts and text transform for preset styles
    styles = {
        'classic' : { 'fonts': [ 'cloistrk','lucian','raleigh'], 'invert':['belwe','raleigh'], 'transform': [cap] },
        'minimal' : { 'fonts': [ 'geometr', 'myriadpro','futura'],'invert':[ 'geometr'],'transform':[upper]},
        'grunge' : { 'fonts': ['distress', 'pantspatrol','polaroid','eklektic'], 'invert':['distress','polaroid'],'transform':[upper,lower]}
    }

    style = styles[form['style']]

    fonts_to_use = np.array(['fonts/'+name+'.ttf' for name in style['fonts']])
    fonts_to_use[np.argwhere(fonts_to_use=="fonts/myriadpro.ttf")] = "fonts/myriadpro.otf"

    if t_f(form['mask']):
        fonts_header = ['fonts/'+name+'.ttf' for name in style['invert']]
    else:
        fonts_header = [font for font in fonts_to_use if font != 'fonts/polaroid.ttf']

    
    #getting processed content from Google
    words, cmap = get_words(form['concept'],form['identifier'],form['keywords'])

    log_progress(form['identifier'],4)
    
    '''
    Running k-means once again to find 'whites' and 'blacks' clusters
    and get rid of one of them depending if background is dark or light
    '''
    color_bins = np.array([[0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],[255,0,255],[0,255,255],[255,255,0]])
    key_colors = k_means(cmap,8,init=color_bins)

    if inverter==-1:
        fin_cmap = np.array(
            [tuple(np.mean(cmap[key_colors[1]==0],axis=0).astype(int))] +
            [ tuple(each) for each in cmap[key_colors[1]!=0] ])
        fin_cmap = fin_cmap[::-1]
    else:
        fin_cmap = np.array(
            [ tuple(each) for each in cmap[key_colors[1]!=1] ] +
            [tuple(np.mean(cmap[key_colors[1]==1],axis=0).astype(int))])

    fin_cmap = [ tuple(each) for each in fin_cmap ]


    #initializing graphic cloud object
    image = Canvas(w,h,round(w/reducer),partial)
    image.fit(form['concept'],fonts_header[np.random.randint(0,len(fonts_header))],invert=t_f(form['mask']))

    drops = []
    stopper = 20 if t_f(form['mask']) else 27

    for num,word in enumerate(words):
        if np.random.randint(0,4)==2:
            if num<2:
                num = 2
            c = str(num) + ' words placed'
            log_progress(form['identifier'],4,message="Mapping... "+c)
        tr = style['transform'][np.random.randint(0,len(style['transform']))]
        drops.append(Droplet(tr(word)))
        drops[-1].fit(fonts_to_use[np.random.randint(0,len(fonts_to_use))])
        #if the mask is very dense cycle can exit prematurely when cycling through central part
        if image.paste_object(drops[-1])<stopper:
            if image.paste_object(drops[-1])<stopper:
                if image.paste_object(drops[-1])<stopper:
                    if image.paste_object(drops[-1])<stopper:
                        log_progress(form['identifier'],4,message="No more free space! Finishing")
                        break

    log_progress(form['identifier'],5)
    
    image.render(fin_cmap)
    image.img.save('static/tmp/'+form['identifier']+'.jpg',quality=90)

    #converting all Image objects to numpy arrays to pickle
    #for element in image.queue[1:]:
    #    element['src'].pickle()
    #image.pickle()

    #pickle.dump( image, open( "pickles/"+form['identifier']+".pkl", "wb" ) )ls

    #creating output dictionary
    concept = 'All about "{}"'.format(form['concept'])
    if form["keywords"]:
        concept += " ({})".format(form["keywords"])

    src = '/tmp/'+form['identifier']+'.jpg'

    caption = "{} style, mapping words {}, {} background<br><a href={}>({}x{})</a>".format(
        cap(form["style"]),
        ('inside' if t_f(form['mask']) else 'outside'),
        ('dark' if inverter==-1 else 'light'),
        src, form['im_width'], form['im_height'])

    #logging request in csv form
    csv = "\t".join([ value for value in form.values() ]) + "\n"
    f=open("requests.csv", "a+")
    f.write(csv)
    f.close()

    return {'concept':concept,'src':src,'caption':caption,'id':form['identifier']}


    