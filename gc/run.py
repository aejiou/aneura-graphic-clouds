import numpy as np
from itertools import cycle

stages = [
    'Getting the links from search engine','Getting content from each of the links',
    'Counting words','Mapping','Rendering the final image']
step = 0

def log_progress(id,message=None):
    global step
    global stages
    f = open("static/tmp/"+id+".log", "w")
    if message==None:
        message = stages[step]
        step += 1
    f.write("Step {} of {}: {}...".format(step,len(stages),message))
    f.close()


def generate_image(form):
    def color_gradient(one,two,steps=10):
        r = np.linspace(one[0],two[0],steps).astype(int)
        g = np.linspace(one[1],two[1],steps).astype(int)
        b = np.linspace(one[2],two[2],steps).astype(int)
        return [ (one,two,three) for one, two, three in zip(r,g,b) ]    

    def upper(w):
        return w.upper()
    def lower(w):
        return w.lower()
    def cap(w):
        return w[:1].upper()+w[1:].lower()
    def rand(w):
        t = [upper,lower,cap][np.random.randint(0,3)]
        return t(w)

    global stages 
    global step   
    step = 0     

    log_progress(form['identifier'])

    t_f = lambda x: True if x=='true' else False

    from src import Canvas, Droplet
    from crawl import get_words


    w, h = int(form['im_width']), int(form['im_height'])

    if t_f(form['mask']): 
        reducer = 1000 
    else:
        reducer = 500

    if t_f(form['inverted']):
        inverter = -1
    else:
        inverter = 1


    fontnames = [
        'antiqua','ashbury','brochurn','cloistrk', 'cushing','distress','eklektic',
        'geometr', 'hobo', 'lucian', 'motterfem', 'myriadpro', 'nuptial', 'pantspatrol',
        'polaroid','raleigh']

    fonts = np.array(['fonts/'+name+'.ttf' for name in fontnames])
    fonts[np.argwhere(fonts=="fonts/myriadpro.ttf")] = "fonts/myriadpro.otf"
    
    fonts_to_use = [t_f(form[name]) for name in fontnames]
    fonts_to_use = fonts[fonts_to_use]

    rainbow = (
        color_gradient((148, 0, 211),(75, 0, 130))[:-1] + 
        color_gradient((75, 0, 130),(0, 0, 255))[:-1] + 
        color_gradient((0, 0, 255),(0, 255, 0))[:-1] + 
        color_gradient((0, 255, 0),(255, 255, 0),4)[:-1] + 
        color_gradient((255, 255, 0),(255, 127, 0))[:-1] + 
        color_gradient((255, 127, 0),(255, 0 , 0))[:-1] + 
        color_gradient((255, 0 , 0),(148, 0, 211))[:-1])
    
    #r = np.random.randint(0,len(rainbow))
    #rainbow = rainbow[r:] + rainbow[:r]
    rainbow = [tuple((np.array(rainbow[0])/2).astype(int))] + rainbow + [(255,255,255)]
    #    [tuple((np.array(rainbow[0])+200).astype(int))])

    colors = {
        'bw': [(0,0,0),(255,255,255)],
        'gray': [(20,20,20)] + color_gradient((100,100,100),(200,200,200),11) + [(230,230,230)],
        'rainbow': rainbow
    }

    image = Canvas(w,h,colors[form['colors']][::inverter],round(w/reducer))
    image.fit(form['name'],fonts_to_use[np.random.randint(0,len(fonts_to_use))],invert=t_f(form['mask']))

    #words = [
    #    'hello','world','all','is','fine','butterflies','unicorns','pagan','rituals',
    #    'horses','cat','no','yes','forever','dark']

    words = get_words(form['name'],form['identifier'])
    
    transform = {'upper':upper,'lower':lower,'cap':cap,'rand':rand}    

    drops = []

    for num,word in enumerate(words):
        if np.random.randint(0,4)==2:
            c = str(num) + ' words placed'
            log_progress(form['identifier'],message="Mapping... "+c)
        drops.append(Droplet(transform[form['transform']](word)))
        drops[-1].fit(fonts_to_use[np.random.randint(0,len(fonts_to_use))])
        if image.paste_object(drops[-1])<6:
            log_progress(form['identifier'],message="No more free space! Finishing")
            break
       

    #for _ in range(0,8):
    #    log_progress(form['identifier'])
    #    for drop in drops:
            

    log_progress(form['identifier'])
    image.render()
    image.img.save('static/tmp/'+form['identifier']+'.jpg',quality=90)

    return {'concept':form['name'],'src':'/tmp/'+form['identifier']+'.jpg','caption':str(form)}


    