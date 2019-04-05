import numpy as np
from itertools import cycle


def generate_image(form):
    def color_gradient(one,two,steps=10):
        r = np.linspace(one[0],two[0],steps).astype(int)
        g = np.linspace(one[1],two[1],steps).astype(int)
        b = np.linspace(one[2],two[2],steps).astype(int)
        return [ (one,two,three) for one, two, three in zip(r,g,b) ]
    
    def log_progress(id):
        nonlocal step
        nonlocal stages
        f = open("static/tmp/"+id+".log", "w")
        f.write("Step {} of {}: {}...".format(step,len(stages),stages[step-1]))
        f.close()
        step += 1

    def upper(w):
        return w.upper()
    def lower(w):
        return w.lower()
    def cap(w):
        return w[:1].upper()+w[1:].lower()
    def rand(w):
        t = [upper,lower,cap][np.random.randint(0,3)]
        return t(w)

    stages = [
        'Creating all the objects',
        'Mapping cycle 1','Mapping cycle 2','Mapping cycle 3','Cycle 4','Cycle 5',
        'Cycle 6','Cycle 7','Cycle 8','Rendering the final image']
    step = 1        
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
    
    r = np.random.randint(0,len(rainbow))

    rainbow = rainbow[r:] + rainbow[:r]
    rainbow = [tuple((np.array(rainbow[0])/2).astype(int))] + rainbow
    if inverter==-1:
        rainbow = [tuple((np.array(rainbow[0])/2).astype(int))] + rainbow
    else:
        rainbow = rainbow + [(255,255,255)]

    colors = {
        'bw': [(0,0,0),(255,255,255)],
        'gray': [(70,70,70)] + color_gradient((100,100,100),(200,200,200),11) + [(230,230,230)],
        'rainbow': rainbow
    }

    test = Canvas(w,h,colors[form['colors']][::inverter],round(w/reducer))
    test.fit(form['name'],"fonts/SCB.TTF",invert=t_f(form['mask']))

    #words = [
    #    'hello','world','all','is','fine','butterflies','unicorns','pagan','rituals',
    #    'horses','cat','no','yes','forever','dark']

    words = get_words(form['name'])
    
    transform = {'upper':upper,'lower':lower,'cap':cap,'rand':rand}    

    drops = []

    for ind,word in enumerate(cycle(words)):
        drops.append(Droplet(transform[form['transform']](word)))
        drops[-1].fit(fonts_to_use[np.random.randint(0,len(fonts_to_use))])
        if test.paste_object(drops[-1])<10:
            break
       

    #for _ in range(0,8):
    #    log_progress(form['identifier'])
    #    for drop in drops:
            

    log_progress(form['identifier'])
    test.render()
    test.img.save('static/tmp/'+form['identifier']+'.jpg',quality=90)

    return {'concept':form['name'],'src':'/tmp/'+form['identifier']+'.jpg','caption':str(form)}


    