
# coding: utf-8

# In[14]:

import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import random
import math

def find_match(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print(des1.shape)
    
    x1_L_R = []
    x2_L_R = []
    nbrs1 = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(des1)
    distances1, indices1 = nbrs1.kneighbors(des2)
    for i in range(0, distances1.shape[0]):
        if distances1[i,0] < (0.7 * distances1[i,1]):  
            x2_L_R.append([kp2[i].pt[0], kp2[i].pt[1]])
            x1_L_R.append([kp1[indices1[i,0]].pt[0], kp1[indices1[i,0]].pt[1]])
        
    x1_R_L = []
    x2_R_L = []
    nbrs2 = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(des2)
    distances2, indices2 = nbrs2.kneighbors(des1)
    for i in range(0, distances2.shape[0]):
        if distances2[i,0] < 0.70*distances2[i,1]:
            x1_R_L.append([kp1[i].pt[0], kp1[i].pt[1]])
            x2_R_L.append([kp2[indices1[i,0]].pt[0], kp2[indices1[i,0]].pt[1]])
    
    x1 = []
    x2 = []
    for i in range(len(x1_L_R)):
        if x1_L_R[i] not in x1:
            x1.append(x1_L_R[i])
            x2.append(x2_L_R[i])
            
    for i in range(len(x1_R_L)):
        if x1_R_L[i] not in x1:
            x1.append(x1_R_L[i])
            x2.append(x1_R_L[i])
            

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    A = [ [0]*3 for i in range(3)]
    max_count = 0 
    
    for i in range(0, ransac_iter):
        indices = np.random.choice(x1.shape[0], 3, replace=False)
        
        x = []
        y = []
        x = [[x1[indices[0],0], x1[indices[0],1], 1, 0, 0, 0], [0, 0, 0, x1[indices[0],0], x1[indices[0],1], 1], [x1[indices[1],0], x1[indices[1],1], 1, 0, 0, 0], [0, 0, 0, x1[indices[1],0], x1[indices[1],1], 1],[x1[indices[2],0], x1[indices[2],1], 1, 0, 0, 0], [0, 0, 0, x1[indices[2],0], x1[indices[2],1], 1] ]
        y = [[x2[indices[0],0]],[x2[indices[0],1]],[x2[indices[1],0]],[x2[indices[1],1]],[x2[indices[2],0]],[x2[indices[2],1]]]
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        x_transpose = np.transpose(x)
        prod = np.dot(x_transpose,x)
        inv = np.linalg.inv(prod)
        b = np.dot(inv,x_transpose)
        at = np.dot(b,y)
       
        inlier_cnt = 0 
        for i in range(0,x1.shape[0]):
            x_1 = np.array([[ x1[i,0], x1[i,1], 1, 0, 0, 0], [ 0, 0, 0, x1[i,0], x1[i,1], 1]])
            x_2 = np.array([[x2[i,0]], [x2[i,1]]])
            x_2_pred = np.dot(x_1,at)
            dis = x_2_pred - x_2
            err = math.hypot(dis[0], dis[1])
            if err < ransac_thr : 
                inlier_cnt = inlier_cnt + 1
        if inlier_cnt > max_count :
            A = at
            max_count = inlier_cnt
            
    
    A = np.array([[A[0,0], A[1,0], A[2,0]], [A[3,0], A[4,0], A[5,0]], [0, 0, 1]])
    
    return A

def warp_image(img, A, output_size):
    img_warped = np.zeros((output_size))
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            ip = np.array([[j], [i], [1]])
            op = np.floor(np.dot(A,ip))
            op = op.astype(int)
            if (op[0] < img.shape[1]) and (op[1] < img.shape[0]):
                img_warped[i,j] = img[op[1], op[0]]
    return img_warped


def align_image(template, target, A):
    p =np.array([[A[0,0]+1, A[0,1], A[0,2]], [A[1,0], A[1,1]+1, A[1,2]], [0, 0, 1]])
    
    filter_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    filter_y = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    gauss_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    
    filter_x = np.matmul(filter_x, gauss_filter)
    filter_y = np.matmul(gauss_filter, filter_y)
    
    im = template
    filter = filter_x
    im_padded = np.pad(im, 1, mode = 'constant')
    im_filtered = np.zeros(im.shape)
    center_k = math.floor(filter.shape[0]/2)+1
    center_l = math.floor(filter.shape[1]/2)+1
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            v = 0
            for k in range(0, filter.shape[0]):
                for l in range(0, filter.shape[1]):
                    i1 = i + k - center_k
                    j1 = j + l - center_l
                    if (i1 <= 0) or (i1 > im_filtered.shape[0]) or (j1 <= 0) or (j1 > im_filtered.shape[1]):
                        continue
                    v = v + im_padded[i1,j1]*filter[k,l]
                
            im_filtered[i,j] = v
    im_dx = im_filtered
    
    filter = filter_y
    im_padded = np.pad(im, 1, mode = 'constant')
    im_filtered = np.zeros(im.shape)
    center_k = math.floor(filter.shape[0]/2)+1
    center_l = math.floor(filter.shape[1]/2)+1
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            v = 0
            for k in range(0, filter.shape[0]):
                for l in range(0, filter.shape[1]):
                    i1 = i + k - center_k
                    j1 = j + l - center_l
                    if (i1 <= 0) or (i1 > im_filtered.shape[0]) or (j1 <= 0) or (j1 > im_filtered.shape[1]):
                        continue
                    v = v + im_padded[i1,j1]*filter[k,l]
                
            im_filtered[i,j] = v
    im_dy = im_filtered
    
    im_dx_f = im_dx.flatten()
    im_dy_f = im_dy.flatten()
    I = np.array([im_dx_f,im_dy_f])
    
    descent_img = np.zeros((template.shape[0]*template.shape[1], 6))
    I = np.transpose(I)
    
    
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            pixel_grad = np.array([I[i*template.shape[1]+j]]).reshape(1,2)
            dw_dp = np.array([[j,i,1,0,0,0],[0,0,0,j,i,1]]) #Jacobian
            descent_img[i*template.shape[1]+j] = pixel_grad @ dw_dp
    
    H = np.transpose(descent_img) @ descent_img
    
    H_inv = np.linalg.inv(H)
    e = 0.03
    d_i_t = np.transpose(descent_img)
    img_err_lst = []
    for k in range(100):
        img_warped = np.zeros(template.shape)
        for i in range(template.shape[0]):
            for j in range(template.shape[1]):
                input_coords = np.array([[j], [i], [1]])
                x2_out = np.floor(np.dot(A,input_coords))
                x2_out = x2_out.astype(int)
                if (x2_out[0] < target.shape[1]) and (x2_out[1] < target.shape[0]):
                    img_warped[i,j] = target[x2_out[1], x2_out[0]]
        img_err = np.subtract(img_warped , template)
        
        i_r = img_err.flatten()
        img_err_lst.append(np.linalg.norm(img_err))
       
        prod = np.array([d_i_t @ i_r]) 
       
        prod = np.transpose(prod)
        del_p = np.dot(H_inv,prod)
       
        del_A = np.array([[del_p[0][0]+1, del_p[1][0], del_p[2][0]], [del_p[3][0], del_p[4][0]+1, del_p[5][0]], [0, 0, 1]])
        
        del_A_inv = np.linalg.inv(del_A)
        A = np.dot(A,del_A_inv)
        mag_del_p = np.linalg.norm(del_p)
        print("Iteration : ",k)
    A_refined = A
    img_err_lst = np.asarray(img_err_lst) / 255
    
    return A_refined, img_err_lst



def track_multi_frames(template, img_list):
    A_list = []
    img1 = img_list[0]
    x1, x2 = find_match(template, img1)
    visualize_find_match(template, img1, x1, x2)
    A = align_image_using_feature(x1, x2, 6, 10000)
    A, errors = align_image(template, img_list[0], A)
    A_list.append(A)
    template = warp_image(img_list[0], A, template.shape)
    for i in range(1,len(img_list)):
        A, errors = align_image(template, img_list[i], A_list[i-1])
        A_list.append(A)
        template = warp_image(img_list[i], A, template.shape)
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)
    ransac_thr = 3
    ransac_iter = 500
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)

