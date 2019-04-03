def generate_image(attr):
    from src import Canvas, Droplet
    test = Canvas(500,250,(0,0,0))
    test.fit('love',"fonts/SCB.TTF")

    colors = [(num*2,num-30,num) for num in range(50,120,10)]

    drops = [Droplet(word,cmap=(color,(0,0,0))) for word,color in zip(['hello','world','all','is','fine','butterflies','unicorns'],colors)]
    drops_u = [Droplet(word.upper(),cmap=(color,(0,0,0))) for word,color in zip(['hello','world','all','is','fine','butterflies','unicorns'],colors)]
    drops += drops_u
    for drop in drops:
        drop.fit("fonts/SCB.TTF")

    for cycle in range(0,5):
        for drop in drops:
            test.paste_object(drop)

    test.img.save('static/tmp/test.jpg')

    return {'concept':'love','src':'/tmp/test.jpg','caption':'blablabla'}

    

    