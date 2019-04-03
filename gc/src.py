from PIL import Image, ImageFilter
from PIL import Image, ImageDraw, ImageFont, ImageOps 
from PIL import ImageFont
from PIL import ImageDraw

import numpy as np
import pandas as pd

from math import floor, ceil


class Canvas:
    def __init__(self,w,h,bg=(0,0,0)):
        self.w = w
        self.h = h
        self.bg = bg
        self.img = Image.new("RGB", (self.w,self.h),bg)
        self.draw = ImageDraw.Draw(self.img)


        
    def fit(self,word,fontfile,filemask=None,margins=(50,50,50,50),invert=False):
        self.word = word
        self.filemask = filemask
        self.margins = margins
        self.invert = invert
        self.mask = Image.new("L", (self.w,self.h),0)

        
        if filemask == None:
            self.dmask = ImageDraw.Draw(self.mask)
            (x, y), size = self._find_fontsize(self.w-(margins[0]+margins[2]),
                                         self.h-(margins[1]+margins[3]),
                                         fontfile,word,showim=False)
            font = ImageFont.truetype(fontfile,size)
            self.dmask.text((x+margins[0], y+margins[1]),word,255,font=font)
        else:
            im = Image.open(filemask).convert('L')
            im.thumbnail(self.mask.size, Image.ANTIALIAS)
            self.mask.paste(im,(round((self.w - im.size[0])/2),round((self.h-im.size[1])/2)))
            
        if invert:
            self.mask = ImageOps.invert(self.mask)
        
        self.th_mat = self._threshold_matrix(self.mask)
        
        return self

        
        
        
    def _find_fontsize(self,w,h,font_url,text,showim=False):
        def get_font_metrics(font_url,text,size):
            font = ImageFont.truetype(font_url,size)
            w, h = font.getsize(text)
            (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
            return (w,h), (offset_x, offset_y)
        def go_search(start,shift):
            nonlocal font_url, text
            start += shift
            (tw,th), (offset_x, offset_y) = get_font_metrics(font_url,text,start)
            aw = tw - offset_x
            ah = th - offset_y
            offset = lambda x: -1 if x else 1
            new_shift = offset((aw>w or ah>h))
            if new_shift != -shift:
                return go_search(start,new_shift)
            else:     
                return (tw + offset_x, th + offset_y, start)

        start = min(w,h)
        x,y, finsize = go_search(start,0)
        return (round((w-x)/2),round((h-y)/2)), finsize
        
    def _threshold_matrix(self,img,th=1):
        pixels = img.load()
        w,h = img.size
        image_matrix = np.zeros((h,w))
        threshold = lambda x: 1 if x>2 else 0
        for col in range(0,w):
            for row in range(0,h):
                image_matrix[row,col]=pixels[col,row]>th
        #if self.invert:
        #    image_matrix = (image_matrix==0)
        return image_matrix.astype(int)
    

    def _density_matrix(self,shape='rect',ratio=(1,1)):
        def _shape(val,shape='rect'):
            nonlocal ratio
            (x,y) = val
            edge = min([x,y,self.h-x,self.w-y])
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
        
        #ratio = np.array(ratio)/max(ratio)

        self.den_mat = np.zeros(self.th_mat.shape)-1
        (h,w) = self.th_mat.shape
        for index in np.ndindex(self.th_mat.shape):
            if self.th_mat[index]==0:
                self.den_mat[index] = _shape(index,shape)
                
    def paste_object(self,obj,where='max'):
        img = obj.src.copy()
        ratio = np.array(obj.ratio)/max(obj.ratio)
        if np.random.randint(0,2)==1:
            ratio = ratio[::-1]
            img = img.rotate(90,Image.NEAREST,True)
        self._density_matrix('rect',ratio)
        maxrad = np.max(self.den_mat)
        maxoptions = np.argwhere(self.den_mat.reshape(-1) == np.max(self.den_mat.reshape(-1)))
        choice = maxoptions[np.random.randint(0,maxoptions.size)]
        maxcenter = np.unravel_index(choice,self.den_mat.shape)
        w, h = floor(ratio[0]*maxrad*2)-2, floor(maxrad*ratio[1]*2)-2
        x, y = int(ceil(maxcenter[1]-maxrad*ratio[0]))+1, int(ceil(maxcenter[0]-maxrad*ratio[1]))+1
        img.thumbnail((w,h), Image.ANTIALIAS)
        self.img.paste(img,(x,y))
        img_g = img.convert('L')
        self.mask.paste(img,(x,y))
        self.th_mat = self._threshold_matrix(self.mask)
        return maxrad


class Droplet:
    def __init__(self,word,bg=(0,0,0),cmap=((255,255,255),(0,0,0)),image=False):
        self.word = word
        self.image = image
        self.bg = bg
        self.cmap = cmap
    
    def fit(self,source):
        if self.image:
            img = Image.open(source).convert('RGB')
            self.w, self.h = img.size
        else:
            font = ImageFont.truetype(source,500)
            tw, th = font.getsize(self.word)
            (width, baseline), (offset_x, offset_y) = font.font.getsize(self.word)
            self.w = tw - offset_x + 20
            self.h = th - offset_y + 20
        
        self.src = Image.new("RGB", (self.w,self.h),self.bg)
        if self.image:
            self.src.paste(img,(0,0))
        else:
            draw = ImageDraw.Draw(self.src)
            draw.text((10-offset_x,10-offset_y),self.word,self.cmap[0],font=font)
            
        self.ratio = (self.w/min(self.w,self.h),self.h/min(self.w,self.h))
        return self


