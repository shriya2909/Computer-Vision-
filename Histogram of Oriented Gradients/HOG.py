
# coding: utf-8

# In[18]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math 

def get_differential_filter():
    #Horizontal diff filter
    filter_x = np.array([[3, 3, 3], [0, 0, 0], [-3, -3, -3]])
    #Vertical diff filter
    filter_y = np.array([[3, 0, -3], [6, 0, -6], [3, 0, -3]])
    #Gaussian filter
    filter_gaus = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    
    #Applying gaussian to diff filter to minimize noise 
    filter_x = np.matmul(filter_x, filter_gaus)
    filter_y = np.matmul(filter_gaus, filter_y)
    
    return filter_x, filter_y


def filter_image(im, filter):
    #Get padded image to apply filter on 
    im_pad = np.pad(im, 1, mode = 'constant')
    #Initialise filtered image
    im_filtered = np.zeros([im.shape[0],im.shape[1]])
    #Apply filter to the padded image(to center appropriately) to get filtered image
    for i in range(1, im.shape[0]):
        for j in range(1, im.shape[1]):
            im_filtered[i,j] = filter[0,0]*im_pad[i-1,j-1] + filter[0,1]*im_pad[i-1,j] + filter[0,2]*im_pad[i-1,j+1] + filter[1,0]*im_pad[i,j-1] + filter[1,1]*im_pad[i,j] + filter[1,2]*im_pad[i,j+1] + filter[2,0]*im_pad[i+1,j-1] + filter[2,1]*im_pad[i+1,j] + filter[2,2]*im_pad[i+1,j+1]
    return im_filtered


def get_gradient(im_dx, im_dy):
    #Get grad_mag : square_root(square of horizontal diff image + square of vertical diff image )
    im_dx_sqr = np.power(im_dx, 2)
    im_dy_sqr = np.power(im_dy, 2)
    sum_sqr =  im_dx_sqr + im_dy_sqr
    grad_mag = np.sqrt(sum_sqr)
    
    #Get grad_angle : tan_inverse(im_dy/im_dx) 
    #Grad_angle returned in degrees
    grad_angle = (np.arctan2(im_dy,im_dx)) * 180 / np.pi
    
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    #Initialise ori_histo to M x N x 6
    #Get M and N : M = floor( grad_mag.rowsize / cell_size ) and N = floor( grad_mag.colsize / cell_size )
    M = math.floor(grad_mag.shape[0]/ cell_size)
    N = math.floor(grad_mag.shape[1]/ cell_size)
    ori_histo = np.zeros((M,N,6))
    
    #Outer 2 loops : iterate over the image such that it divides the image into cells
    #Inner 2 loops : iterate over each cell to retrieve pixels in that cell 
    #Extract pixel angle to decide bin (done via if statements inside the loops)
    #Extract pixel mag and add value to appropriate bin for corresponding cell(inside loop)
    #Update ori_histo (inside loop)
    #Repeat for all cells 
    
    ci = 0 
    cj = 0 
    for i in range(0, (grad_mag.shape[0]- cell_size), cell_size):
        for j in range(0, (grad_mag.shape[0] - cell_size), cell_size):
            for r in range(i, (i + cell_size)):
                for s in range(j, (j + cell_size)):
                    mag = grad_mag[r][s]
                    angle = grad_angle[r][s]
                    if ( angle >= 165 and angle < 180 ) or ( angle >= 0  and angle < 15):
                        ori_histo[ci,cj,0] = ori_histo[ci,cj,0] + mag 
                    elif angle >= 15 and  angle < 45 :
                        ori_histo[ci,cj,1] = ori_histo[ci,cj,1] + mag 
                    elif angle >= 45 and angle < 75 :
                        ori_histo[ci,cj,2] = ori_histo[ci,cj,2] + mag 
                    elif angle >= 75 and angle < 105:
                        ori_histo[ci,cj,3] = ori_histo[ci,cj,3] + mag 
                    elif angle >= 105 and angle < 135:
                        ori_histo[ci,cj,4] = ori_histo[ci,cj,4] + mag 
                    else:
                        ori_histo[ci,cj,5] = ori_histo[ci,cj,5] + mag 
            cj = cj + 1
        ci = ci + 1
        cj = 0

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    #Initialise ori_histo_normalized to [M - (block_size - 1)] x [N - (block_size - 1)] x [6 * (block_size ^ 2)]  
    x = block_size - 1
    M1 = ori_histo.shape[0] - x
    N1 = ori_histo.shape[1] - x
    block_size_sqr = block_size * block_size 
    y = 6 * block_size_sqr
    ori_histo_normalized = np.zeros((M1,N1,y))
    
    #Outer 2 loops : iterate over the image such that it divides the image into blocks
    #Initialise vector for that block 
    #Inner 2 loops : iterate over each blocks to retrieve cells in that block 
    #Inner most loop : iterate over the 6 bins for each cell of that block 
    #Get each bin value for the cells and stack them into the long vector for that block (Inside Inner loops)
    #Calculate sum of squares for the block(Inside Outer loops)
    #Divide all the values in the long vector by the sum of squares(Inside Outer loops)v
    #Repeat for all blocks 
    
    for i in range(0, ori_histo.shape[0]-block_size):
        for j in range(0, ori_histo.shape[0]-block_size):
            vb = np.zeros((block_size_sqr*6,1))
            bc = 0
            ss = 0
            for r in range(i,i+block_size):
                for s in range(j, j+block_size):
                    for k in range(0,6):
                        vb[bc,0] = ori_histo[r,s,k]
                        bc = bc + 1
            for x in range(0, vb.shape[0]):
                ss = ss + np.power(vb[x],2)
            for x in range(0, vb.shape[0]):
                ori_histo_normalized[i,j,x] = vb[x]/(np.sqrt(ss + np.power(0.001,2)))
    
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    filter_x, filter_y = get_differential_filter()

    horizontalFiltered = filter_image(im, filter_x)
    horizontalFiltered = horizontalFiltered.astype(np.uint8)
    plt.imshow(horizontalFiltered)
    plt.show()
    cv2.imwrite('horizontal.jpg', horizontalFiltered)
    
    verticalFiltered = filter_image(im, filter_y)
    verticalFiltered = verticalFiltered.astype(np.uint8)
    plt.imshow(verticalFiltered)
    plt.show()
    cv2.imwrite('vertical.jpg', verticalFiltered)

    grad_magnitude, grad_angle = get_gradient(horizontalFiltered, verticalFiltered)
    plt.imshow(grad_angle.astype(np.uint8))
    plt.show()
    cv2.imwrite('grad_angle.jpg', grad_angle.astype(np.uint8))
    plt.imshow(grad_magnitude.astype(np.uint8))
    plt.show()
    cv2.imwrite('grad_magnitude.jpg', grad_magnitude.astype(np.uint8))
    
    ori_histo = build_histogram(grad_magnitude, grad_angle, 8)
    visualize_hog_cell(im,ori_histo,8)
    
    ori_histo_normalised = get_block_descriptor(ori_histo,2)
    hog = ori_histo_normalised.flatten()
    visualize_hog_block(im,ori_histo_normalised,8,2)
    
    return hog

def visualize_hog_block(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7 # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[int(cell_size*block_size/2): cell_size*num_cell_w-(cell_size*block_size/2)+1: cell_size], np.r_[int(cell_size*block_size/2): cell_size*num_cell_h-(cell_size*block_size/2)+1: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i], color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

def visualize_hog_cell(im, ori_histo, cell_size):
    norm_constant = 1e-3
    num_cell_h, num_cell_w, num_bins = ori_histo.shape
    max_len = cell_size / 3
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size/2: cell_size*num_cell_w: cell_size], np.r_[cell_size/2: cell_size*num_cell_h: cell_size])
    bin_ave = np.sqrt(np.sum(ori_histo ** 2, axis=2) + norm_constant ** 2) # (ori_histo.shape[0], ori_histo.shape[1])
    histo_normalized = ori_histo / np.expand_dims(bin_ave, axis=2) * max_len # same dims as ori_histo
    mesh_u = histo_normalized * np.sin(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    mesh_v = histo_normalized * -np.cos(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - mesh_u[:, :, i], mesh_y - mesh_v[:, :, i], 2 * mesh_u[:, :, i], 2 * mesh_v[:, :, i],
        color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()



if __name__=='__main__':
    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)

