from PIL import Image, ImageFilter
from PIL import ImageDraw, ImageFont, ImageOps

import numpy as np

from math import floor, ceil

def find_fontsize(w,h,font_url,text):
    def get_font_metrics(font_url,text,size):
        font = ImageFont.truetype(font_url,size)
        w, h = font.getsize(text)
        (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
        return (w,h), (offset_x, offset_y)
    def go_search(start,shift,step):
        nonlocal font_url, text
        start += shift
        (tw,th), (offset_x, offset_y) = get_font_metrics(font_url,text,start)
        aw = tw - offset_x
        ah = th - offset_y
        offset = lambda x: -1 if x else 1
        new_shift = offset((aw>w or ah>h))*step
        if new_shift != -shift:
            return go_search(start,new_shift,step)
        else:   
            if step !=1:
                return go_search(start,round(new_shift/step),1)
            else:  
                return tw + offset_x, th + offset_y, start

    start = min(w,h)
    x,y, finsize = go_search(start,0,10)
    return (round((w-x)/2),round((h-y)/2)), finsize

def monochrome(img,th=127):
    img = img.convert("L")
    return Image.fromarray(((np.asarray(img)>th)*255).astype('uint8'))


class Canvas:
    def __init__(self,w,h,reduce=4,partial=(1,1)):
        '''
        Saving dimentions of the image: w,h, colormap: cmap,
        by how many times is mask smaller than the image: reduce,
        which portion of the mask to use each time when looking 
        for the best place to place an object: partial.
        Then an empty image and queue of objects are created.
        '''
        self.partial = partial                
        self.reduce = reduce
        self.w = w
        self.h = h
        self.queue = []
        self.cmap = None

        
    def fit(self,word,fontfile,filemask=None,margins=(50,50,50,50),invert=False):
        '''
        The central concept: word, font for it: fontfile, 
        alpha mask as a file:filemask, space to the edge: margins,
        whether to place objects inside the mask: invert
        Mask is initialized as black greyscale canvas and 
        word is placed in the middle.
        Array of slice positions and size is initialized
        based on partial.
        Threshold matrix based on mask is created.
        Array containing indexes of each pixel is created.
        '''
        self.word = word
        self.filemask = filemask
        self.margins = margins
        self.m_margins = np.round(np.array(margins)/self.reduce).astype(int)
        self.invert = invert
        self.mask_h = floor((self.h/self.reduce))
        self.mask_w = floor((self.w/self.reduce))        
        self.mask = Image.new("L", (self.mask_w,self.mask_h),0)

        
        if filemask == None:
            #draw = ImageDraw.Draw(self.mask)
            #(x, y), size = find_fontsize(self.mask_w-(self.m_margins[0]+self.m_margins[2]),
            #                             self.mask_h-(self.m_margins[1]+self.m_margins[3]),
            #                             fontfile,
            #                             word)
            (x, y), size = find_fontsize(self.w-(self.margins[0]+self.margins[2]),
                                         self.h-(self.margins[1]+self.margins[3]),
                                         fontfile,
                                         word)
            font = ImageFont.truetype(fontfile,size)
            tmp = Image.new("L", (self.w,self.h),0)
            draw = ImageDraw.Draw(tmp)
            draw.text((x+self.margins[0], y+self.margins[1]),word,255,font=font)
            self.mask.paste(tmp.resize(self.mask.size,Image.ANTIALIAS))
            self.queue.append({'src':'font','args':{'file':fontfile,'pos':(x, y),'size':size,'color':None}})
        else:
            im = Image.open(filemask).convert('L').resize(self.mask.size, Image.ANTIALIAS)
            self.mask.paste(im,(round((self.mask_w - im.size[0])/2),round((self.mask_h-im.size[1])/2)))
            self.queue.append({'src':'image',
                               'args':{'file':filemask,'color':None}})

        s_num = [0 if each==1 else ceil(1/each) for each in self.partial]    
        self.slices = []
        self.slice_size = ( floor(self.mask_h*self.partial[1]),floor(self.mask_w*self.partial[0]) )
        step = [(1-p)/n if n>0 else 1 for p,n in zip(self.partial,s_num)]
        for i in range(0,s_num[1]+1):
            for j in range(0,s_num[0]+1):
                self.slices.append([floor(i*step[1]*self.mask_h),floor(j*step[0]*self.mask_w)])
        self.current_slice = 0

            
        if invert:
            self.mask = ImageOps.invert(self.mask)
        
        self.th_mat = self._threshold_matrix(self.mask)
        
        self.ind_arr = np.zeros(self.th_mat.shape).astype(str)
        for index in np.ndindex(self.th_mat.shape):
            self.ind_arr[index] = "{} {}".format(*index)
        
        return self

        
    def _threshold_matrix(self,img,th=1):
        '''
        Returns matrix of 0 and 1 based on greyscale image.
        1 for white, white is where you can't place new
        objects.
        '''
        return (np.array(img)>th).astype(int)    

    def _density_matrix(self,offset,size,shape='rect',ratio=(1,1)):
        '''
        Calculating which size shape can be placed
        at each of the possible positions according to
        the mask. Deprecated.
        '''
        def _shape(val,shape='rect'):
            nonlocal ratio
            (x,y) = val
            edge = min([x,y,self.mask_h-x,self.mask_w-y])
            if edge==0:
                rad = 1
            else:
                rad = int(self.den_mat[x-1,y-1]-1)
            while rad < edge:
                xr = ceil(rad*ratio[1])
                yr = ceil(rad*ratio[0])
                if np.sum(self.th_mat[x-xr:x+1+xr,y-yr:y+1+yr])==0:
                    rad +=1
                else:
                    break
            return int(rad)

        (offset_y, offset_x) = offset
        (h, w) = size
        self.den_mat = np.zeros(self.th_mat.shape)-1
        self.den_mat[offset_y:offset_y + h,offset_x:offset_x + w] = -self.th_mat[offset_y:offset_y + h,offset_x:offset_x + w]
        for index_str in self.ind_arr[self.den_mat!=-1]:
            index = [int(_) for _ in index_str.split(' ')]
            self.den_mat[index[0]][index[1]] = _shape(index,shape)
                        
    def paste_object(self,obj,where='max'):
        '''
        Placing each of the objects in the best position
        on the mask, saving the coordinates into the queue.
        Max radius is returned, if it's very small then
        there's likely no free space.
        '''
        img = obj.mask.copy()
        ratio = np.array(obj.ratio)/max(obj.ratio)
        if np.random.randint(0,2)==1:
            rotate = True
            ratio = ratio[::-1]
            img = img.rotate(90,Image.NEAREST,True)
        else:
            rotate = False
        self._density_matrix(self.slices[self.current_slice],self.slice_size,'rect',ratio)
        self.current_slice = self.current_slice + 1 if self.current_slice<len(self.slices)-1 else 0
        maxrad = np.max(self.den_mat)
        maxoptions = np.argwhere(self.den_mat.reshape(-1) == np.max(self.den_mat.reshape(-1)))
        choice = maxoptions[np.random.randint(0,maxoptions.size)]
        maxcenter = np.unravel_index(choice,self.den_mat.shape)
        w, h = ratio[0]*(maxrad*2-2), (maxrad*2-2)*ratio[1]
        x, y = maxcenter[1]-maxrad*ratio[0]+1, maxcenter[0]-maxrad*ratio[1]+1
        w, h = floor(w), floor(h)
        x, y = ceil(x), ceil(y)

        if min(w,h)<0.5:
            if max(w,h)>7:
                return 15*self.reduce, False
            else:
                return 0, False

        self.queue.append({'src':obj,'args':{'size':(w*self.reduce,h*self.reduce),
                                             'pos':(x*self.reduce,y*self.reduce), 
                                             'rotate':rotate,
                                             'color': None}})
        img = img.resize((w,h), Image.ANTIALIAS)
        
        self.th_mat[y:y+img.size[1],x:x+img.size[0]] = self._threshold_matrix(img)
        self.mask.paste(img,(x,y))
        self.th_mat = self._threshold_matrix(self.mask)
        return maxrad*self.reduce, True
        
    def render(self,cmap=None,size=None, margins=(0,0,0,0),alpha=True):
        '''
        Rendering the final hi-res image from the queue
        and appying alpha gaussian blur.
        '''   
        if self.cmap != cmap and cmap!=None:
            if self.cmap==None:
                self.cmap = cmap
            new_colors = True
            bg = cmap[-1]
            if self.queue[0]['args']['color'] == None:
                self.queue[0]['args']['color'] = self.cmap[0]
                self.queue[0]['args']['bg'] = bg
        else:
            new_colors = False
            if self.cmap == None:
                return 'Initialize with a color map first'
            else:
                cmap = self.cmap
                bg = self.queue[0]['args']['bg']

        if size == None:
            size = (self.w,self.h)

        imgsize = (size[0]-margins[0]-margins[2],size[1]-margins[1]-margins[3])
        scale = min( imgsize[0]/self.w, imgsize[1]/self.h )
        imgsize = ( int(scale*self.w),int(scale*self.h) )

        self.img = Image.new("RGB", imgsize,bg)

        for element in self.queue[1:]:
            if isinstance(element['src'],Droplet):
                if new_colors:
                    r_color = np.random.randint(0,len(cmap)-1)
                    if element['args']['color'] == None:
                        element['args']['color'] = r_color
                else:
                    r_color = element['args']['color']
                img = element['src'].render((cmap[r_color],bg))
                if element['args']['rotate']:
                    img = img.rotate(90,Image.NEAREST,True)
                w, h = element['args']['size'][0]*scale, element['args']['size'][1]*scale
                x, y = element['args']['pos'][0]*scale, element['args']['pos'][1]*scale
                img = img.resize((int(w),int(h)), Image.ANTIALIAS)
                self.img.paste(img,(int(x),int(y)))

        if self.queue[0]['src']=='font':
            if self.invert==False:
                r_color = cmap[0] if new_colors else self.queue[0]['args']['color']
                if scale==1:
                    (x, y), fsize = self.queue[0]['args']['pos'],self.queue[0]['args']['size']
                    m1, m2 = self.margins[0], self.margins[1]
                else:
                    w = imgsize[0]-(self.m_margins[0]+self.m_margins[2])*self.reduce*scale
                    h = imgsize[1]-(self.m_margins[1]+self.m_margins[3])*self.reduce*scale
                    (x, y), fsize = find_fontsize(int(w),int(h),self.queue[0]['args']['file'],self.word)
                    m1, m2 = self.m_margins[0]*self.reduce*scale, self.m_margins[1]*self.reduce*scale
                font = ImageFont.truetype(self.queue[0]['args']['file'],fsize)
                draw = ImageDraw.Draw(self.img)
                draw.text( (int(x+m1), int(y+m2)), self.word,r_color,font=font)

        if alpha:
            self.img = self._alpha_effect(scale).convert('RGB')

        if margins!=(0,0,0,0):
            new = Image.new("RGB", size, cmap[-1])
            margins_adjust = (int((size[0]-imgsize[0]-margins[0]-margins[2])/2) , int((size[1]-imgsize[1]-margins[1]-margins[3])/2))
            new.paste(self.img,(margins[0]+margins_adjust[0],margins[1]+margins_adjust[1]))
            self.img = new


    def _alpha_effect(self,scale=1):
        '''
        Nice looking blur overlay.
        '''
        bmf = np.array(self.mask.copy().resize(self.img.size,Image.BILINEAR).filter(ImageFilter.GaussianBlur(5*self.reduce*scale)))
        img_alpha = np.array(self.img.convert("RGBA"))
        img_alpha[:,:,3] = bmf
        new_alpha = Image.fromarray(img_alpha)
        result = Image.new("RGBA",self.img.size,(*self.cmap[-1],255))
        for _ in range(0,3):
            result.alpha_composite(new_alpha)
        if self.invert:
            result.alpha_composite(new_alpha)
        return result

    def pickle(self):
        #self.np_img = np.array(self.img)
        self.img = None
        self.np_mask = np.array(self.mask)
        self.mask = None

    def unpickle(self):
        #self.img = Image.fromarray(self.np_img)
        #self.np_img = None
        self.mask = Image.fromarray(self.np_mask)
        self.np_mask = None




class Droplet:
    def __init__(self,word,image=False):
        self.word = word
        self.image = image
    
    def fit(self,source):
        self.file = source
        
        if self.image:
            self.img = Image.open(source)
            self.mask = monochrome(self.img)
            self.w, self.h = self.img.size
        else:    
            font = ImageFont.truetype(self.file,500)
            tw, th = font.getsize(self.word)
            (width, baseline), (self.offset_x, self.offset_y) = font.font.getsize(self.word)
            self.w = tw - self.offset_x + 4
            self.h = th - self.offset_y + 4
        
        self.mask = Image.new("L", (self.w,self.h),0)
        
        if self.image:
            self.mask.paste(self.img,(0,0))
        else:
            draw = ImageDraw.Draw(self.mask)
            draw.text((2-self.offset_x,2-self.offset_y),self.word,255,font=font)
            
        self.ratio = (self.w/min(self.w,self.h),self.h/min(self.w,self.h))
        return self
    
    def render(self,cmap=((255,255,255),(0,0,0)),size=None):
            
        if self.image:
            result = self.img.copy()
            if size!=None:
                result = result.resize((int(size[0]),int(size[1])),Image.ANTIALIAS)
        else:
            if size==None:
                result = Image.new("RGB", (self.w,self.h), cmap[-1])
                draw = ImageDraw.Draw(result)
                font = ImageFont.truetype(self.file,500)
                draw.text((2-self.offset_x,2-self.offset_y),self.word,cmap[0],font=font)
            else:
                (w,h) = (size[0],size[1])
                result = Image.new("RGB", (w,h), cmap[-1])
                draw = ImageDraw.Draw(result)
                (x, y), size = find_fontsize((w-4),(h-4),self.file,self.word)
                draw.text((x+2, y+2),self.word,cmap[0],font=ImageFont.truetype(self.file,size))
            #self.img = result                
                          
        return result
    
    def pickle(self):
        #self.np_img = np.array(self.img)
        self.img = None
        self.np_mask = np.array(self.mask)
        self.mask = None

    def unpickle(self):
        #self.img = Image.fromarray(self.np_img)
        #self.np_img = None
        self.mask = Image.fromarray(self.np_mask)
        self.np_mask = None

    

