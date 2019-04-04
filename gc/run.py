import numpy as np


def generate_image(form):
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

    colors = {
        'bw': [(0,0,0),(255,255,255)],
        'gray': [ 
            tuple(np.vstack([np.zeros(3),(np.zeros(3) + 
            np.arange(60+50*inverter,200+50*inverter,5)[:, None])])[num].astype(int))
            for num in range(0,28) ],
        'rainbow': [
            (75, 0, 130),(148, 0, 211),(0, 0, 255),(0, 255, 0),
            (255, 255, 0),(255, 127, 0),(255, 0 , 0),(255,255,255)]
    }

    test = Canvas(w,h,colors[form['colors']][::inverter],round(w/reducer))
    test.fit(form['name'],"fonts/SCB.TTF",invert=t_f(form['mask']))

    words = [
        'hello','world','all','is','fine','butterflies','unicorns','pagan','rituals',
        'horses','cat','no','yes','forever','dark']
    
    transform = {'upper':upper,'lower':lower,'cap':cap,'rand':rand}    

    drops = [Droplet(transform[form['transform']](word)) for word in words]

    for drop in drops:
        drop.fit(fonts_to_use[np.random.randint(0,len(fonts_to_use))])

    for _ in range(0,8):
        log_progress(form['identifier'])
        for drop in drops:
            test.paste_object(drop)

    log_progress(form['identifier'])
    test.render()
    test.img.save('static/tmp/'+form['identifier']+'.jpg',quality=90)

    return {'concept':form['name'],'src':'/tmp/'+form['identifier']+'.jpg','caption':str(form)}


    