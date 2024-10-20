import torch
from torchvision import transforms
import os
from PIL import Image

class Renderer():
    
    def __init__(self, imW, imH, texPatchSize): #image Width, image Height
        self.imW = imW
        self.imH = imH
        self.texPatchSize = texPatchSize
        
        self.indSrc = torch.arange(imH * imW).view(imH, imW).transpose(1,0).reshape(-1).cuda()
        
    def render_texture_view(self, tex, indTex, indIm, weights): # Texture Index, Image Index 
        # Gather the values required from the texture, then distribute them to the correct pixels in the image.
        # Repeat this four times, once for each part of the bilinear interpolation.
        # Multiply each part with the weights for these pixels
        numChannels = tex.shape[0]
        indTex, indIm = indTex.repeat(numChannels, 1, 1), indIm.repeat(numChannels, 1)
        scattered_weighted = []
        for i in range(4):
            gathered = torch.gather(tex, 1, indTex[:, i, :])
            empty = torch.zeros((numChannels, self.imH * self.imW)).cuda()
            scattered = empty.scatter(1, indIm, gathered).view(numChannels, self.imH, self.imW)
            scattered = scattered * weights[i]
            scattered_weighted.append(scattered)
            
        # Then sum up the parts to create the final image
        return sum(scattered_weighted)
    
    def render(self, tex, indTexList, indImList, weightList):
        return sum([self.render_texture_view(tex, indTex, indIm, weights) 
                                for indTex, indIm, weights 
                                in zip(indTexList, indImList, weightList)
                                ])
        
    def render_batch(self, texBatch, corrBatch):
        
        batch = []
        
        if texBatch.dim() == 3:
            for tex, (indTex, indIm, weights) in zip(texBatch, corrBatch):
                img = self.render(tex, indTex, indIm, weights)
                batch.append(img)
        elif texBatch.dim() == 2:
              