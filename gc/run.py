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
        'Saving the final image']
    #global step 
    step = 1

    log_progress(form['identifier'])

    test = Canvas(int(form['im_width']),int(form['im_height']),(0,0,0))
    test.fit(form['name'],"fonts/SCB.TTF")

    colors = [(num*2,num-30,num) for num in range(50,120,10)]

    drops = [Droplet(word,cmap=(color,(0,0,0))) for word,color in zip(['hello','world','all','is','fine','butterflies','unicorns'],colors)]
    drops_u = [Droplet(word.upper(),cmap=(color,(0,0,0))) for word,color in zip(['hello','world','all','is','fine','butterflies','unicorns'],colors)]
    drops += drops_u
    for drop in drops:
        drop.fit("fonts/SCB.TTF")

    for _ in range(0,5):
        log_progress(form['identifier'])
        for drop in drops:
            test.paste_object(drop)

    log_progress(form['identifier'])
    test.img.save('static/tmp/'+form['identifier']+'.jpg')

    return {'concept':form['name'],'src':'/tmp/'+form['identifier']+'.jpg','caption':str(form)}


    