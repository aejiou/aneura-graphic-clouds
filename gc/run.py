def generate_image(form):
    def log_progress(id):
        nonlocal step
        nonlocal stages
        f = open("static/tmp/"+id+".log", "w")
        f.write("Step {} of {}: {}...".format(step,len(stages),stages[step-1]))
        f.close()
        step += 1

    from src import Canvas, Droplet
    #global stages
    stages = [
        'Creating all the objects',
        'Mapping cycle 1','Mapping cycle 2','Mapping cycle 3','Cycle 4','Cycle 5',
        'Rendering the final image']
    #global step 
    step = 1

    log_progress(form['identifier'])

    colors = [(num*2,num,num) for num in range(50,120,10)]
    colors = [(0,0,0)] + colors + [(255,255,255)]

    w, h = int(form['im_width']), int(form['im_height'])

    test = Canvas(w,h,colors,round(w/500))
    test.fit(form['name'],"fonts/SCB.TTF")

    drops = [Droplet(word) for word in ['hello','world','all','is','fine','butterflies','unicorns']]
    drops_u = [Droplet(word.upper()) for word in ['hello','world','all','is','fine','butterflies','unicorns']]
    
    drops += drops_u
    for drop in drops:
        drop.fit("fonts/SCB.TTF")

    for _ in range(0,5):
        log_progress(form['identifier'])
        for drop in drops:
            test.paste_object(drop)

    log_progress(form['identifier'])
    test.render()
    test.img.save('static/tmp/'+form['identifier']+'.jpg')

    return {'concept':form['name'],'src':'/tmp/'+form['identifier']+'.jpg','caption':str(form)}


    