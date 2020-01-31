import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    n = im_train.shape[1]
    if n % batch_size == 0:
        num_batches = n/batch_size
    else:
        num_batches = np.floor(n/batch_size)+1
    
    mini_batch_x = []
    mini_batch_y = []
    encoding_label = np.zeros((10, n))

    for i in range(0, n):
        encoding_label[label_train[0, i], i] = 1
    perm = np.random.permutation(n)

    for i in range(0, int(num_batches)):
        list_per_batch = im_train[:, perm[i*batch_size:i*batch_size+batch_size]]
        label_per_batch = encoding_label[:, perm[i*batch_size:i*batch_size+batch_size]]
        mini_batch_x.append(list_per_batch)
        mini_batch_y.append(label_per_batch)


    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    #print("fc x ", x.shape)
#    print("fc w ", w.shape)
#    print("fc b ", b.shape)
    #x = np.reshape(x,x.shape)
    y = np.dot(w,x)+b
#    print("fc y output", y.shape)
    return y


def fc_backward(dl_dy, x, w, b, y):
#    print("fc back dl_dy ", dl_dy.shape)
#    print("fc back x ", x.shape)
#    print("fc back w ", w.shape)
#    print("fc back b",b.shape)
#    print("fc back y ", y.shape)
    dl_db = dl_dy
    dl_dx = np.dot(w.T, dl_dy)
    dl_dw = np.dot(dl_dy, x.T)
    
#    print("fc back output dl_dx ", dl_dx.shape)
#    print("fc back output dl_dw ",dl_dw.shape)
#    print("fc back output dl_db", dl_db.shape)
    
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l = np.square(np.linalg.norm( y - y_tilde))
    dl_dy = - (2*(y - y_tilde))
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
#    print("loss_cross_entropy_softmax x",x.shape)
#    print("loss_cross_entropy_softmax y",y.shape)
    softmax = np.exp(x) / np.sum(np.exp(x), axis=0)
    l = np.dot(y.T, np.log(softmax))
    dl_dy = softmax-y
#    
#    print("loss_cross_entropy_softmax output l ",l)
#    print("loss_cross_entropy_softmax output dl_dy" , dl_dy.shape)
#    
    return l, dl_dy

def relu(x):
#    print("Rel x" , x.shape)
    y = np.maximum(0.01*x, x)
#    print("Rel y output ", y.shape)
    return y


def relu_backward(dl_dy, x, y):
#    print("Rel back dl_dy ", dl_dy.shape)
#    print("Rel back y ", y.shape)
#    print("Rel back x ", x.shape)
    dl_dx = (y > 0)*dl_dy
#    print("Rel back output dl_dx ", dl_dx.shape)
    return dl_dx



def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
    # Parameters
    A = A.reshape((A.shape[0], A.shape[1]))
    M,N = A.shape
    col_extent = N - BSZ[1] + 1
    row_extent = M - BSZ[0] + 1

    # Get Starting block indices
    start_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take (A,start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize])


def col2im_sliding(B, block_size, image_size):
    m,n = block_size
    mm,nn = image_size
    return B.reshape(nn-n+1,mm-m+1).T 

def conv(x, w_conv, b_conv):
    x_height, x_width, x_C1 = x.shape
#    print("Conv x " , x.shape)
    w_height, w_width, w_C1, w_C2 = w_conv.shape
#    print("Conv w_conv ",w_conv.shape)
    
    y = np.zeros((x_height, x_width, w_C2))
    x = np.pad(x,((1,1),(1,1),(0,0)), mode = 'constant')
    
    x_col = im2col_sliding_broadcasting(x, [3,3], stepsize=1)
    w_new =np.reshape(w_conv, ( w_height*w_width*w_C1, w_C2))
    #print("w_new", w_new)
    #print("X_col shape", x_col.shape)
    #print("w_new shape", w_new.shape)
    y_col = x_col.T @ w_new
    #print("Y: ",y)
    #print("y.shape before bias", y.shape)

    for i in range(w_C2):
            y[:,:,i] = col2im_sliding(y_col[:,0], [3,3], [x.shape[0], x.shape[1]])
    #print("b_conv: ",b_conv)
    for i in range(0,w_C2):
        y[:,:,i] = y[:,:,i] + b_conv[i][0]
    #print("Y: ",y)
    #print("y.shape after bias", y.shape)
    #y = col2im(y, H, W, C2)
    #print("y.shape after col2im", y.shape)

#    print("Conv padded x ", x.shape)
#    #depth loop - w_C2
#    for d in range(w_C2):
#        #for row in the image 
#        conv_kernel = w_conv[:, :, :, d]
#        conv_kernel = conv_kernel.reshape(w_height, w_width) #3X3
#        for r in range(x_height):
#            #for each col in the image 
#            for c in range(x_width):
#                img_convolve = x[r:r+w_height, c:c+w_width,:]
#                #print(img_convolve.shape)
#                img_convolve = img_convolve.reshape(w_height, w_width) #3X3
#                y[r,c,d]=np.sum(np.sum(np.dot(img_convolve , conv_kernel)))+b_conv[d] 
#
#    print("Conv output y ", y.shape)
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    H,W,C1 = x.shape
    h,w,C1,C2 =w_conv.shape
    #print("x.shape", x.shape)
    x = np.pad(x, ((h//2,h//2), (w//2, w//2), (0,0)), 'constant')
    x_col = im2col_sliding_broadcasting(x, [3,3], stepsize=1)
    #print("dl_dy shape", dl_dy.shape, "x_col shape", x_col.shape)
    dl_dy_new = np.reshape(dl_dy, (H*W,C2))
   
    #print("dl_dy_new shape", dl_dy_new.shape)
    dl_dw_temp = x_col @ dl_dy_new
   
    dl_dw = np.reshape(dl_dw_temp.T, (h,w,C1,C2))
    #print("np.sum(dl_dy, axis=0", np.sum(dl_dy, axis=1).shape)
    dl_db = np.sum(np.sum(dl_dy, axis=0), axis = 0).reshape(b_conv.shape)
    #print("dl_dw shape", dl_dw.shape, "dl_db shape", dl_db.shape)
#    x_height, x_width, x_C1 = x.shape
#    print("Conv back X", x.shape)
#    
#    w_height, w_width, w_C1, w_C2 = w_conv.shape
#    print("Conv back w_conv", w_conv.shape)
#    
#    y_height, y_width, y_C2 = y.shape
#    
#    
#    dl_dy_height, dl_dy_width, dl_dy_C2 = dl_dy.shape
#    
#    
#    print("Conv back b_conv", b_conv.shape)
#    print("Conv back y", y.shape)
#    print("Conv back dl_dy", dl_dy.shape)
#    
#    x = np.pad(x,((1,1),(1,1),(0,0)), mode = 'constant')
#    
#    dl_dw = np.zeros(w_conv.shape)
#    dl_db = np.zeros(b_conv.shape)
#    
#    #CALCULATE dw
#    
#    #depth loop - w_C2
#    for d in range(dl_dy_C2):
#        #for row in the image 
#        for r in range(dl_dy_height):
#            #for each col in the image 
#            for c in range(dl_dy_width):
#                dl_dy_k = dl_dy[r, c,d]
#                x_k = x[r:r+w_height, c:c+w_width,:]
#                prod =np.dot( dl_dy_k , x_k)
#                dl_dw[:,:,:,d] += prod 
#    
#     #CALCULATE db
#   
#    for d in range(dl_dy_C2):
#         dl_db[d] = np.sum(dl_dy[:,:,d])
#         
#    print("Conv back output dl_dw", dl_dw.shape)
#    print("Conv back output dl_db", dl_db.shape)
    return dl_dw, dl_db

def pool2x2(x):
#    print("Pool back x : ",x.shape)
    x_row = x.shape[0]
    x_col = x.shape[1]
    x_C = x.shape[2]
    stride = 2
    y = np.zeros((int(x_row/stride), int(x_col/stride), x_C))
    
    for i in range(0, x_row, stride):
        for j in range(0, x_col, stride):
            for k in range(0, x_C):
                #get block 
                b = x[i: i+stride, j:j+stride, k]
                #get max val for the block 
                mx = b.max()
                #add to output
                y[int((i+stride-1)/2), int((j+stride-1)/2), k] = mx
#    print("Pool back output y : ", y.shape)           
    return y

def pool2x2_backward(dl_dy, x, y):
   
    x_row, x_col, x_C = x.shape
#    print("Pool 2x2 back x : ", x.shape)
#    print("Pool 2x2 back y : ", y.shape)
#    print("Pool 2x2 back dl_dy : ", dl_dy.shape)
    dl_dx = np.zeros(x.shape)
    stride = 2
    for i in range(0, x_row, stride):
        for j in range(0, x_col, stride):
            for k in range(0, x_C):
                if x[i, j, k] == y[int((i+stride-1)/2),int( (j+stride-1)/2), k]:
                    dl_dx[i, j, k] = dl_dy[int((i+stride-1)/2), int((j+stride-1)/2), k]
                elif x[i, int((j+stride-1)/2), k] == y[int((i+stride-1)/2), int((j+stride-1)/2), k]:
                    dl_dx[i, int((j+stride-1)/2), k] = dl_dy[int((i+stride-1)/2),int( (j+stride-1)/2), k]
                elif x[int((i+stride-1)/2), j, k] == y[int((i+stride-1)/2), int((j+stride-1)/2), k]:
                    dl_dx[int((i+stride-1)/2), j, k] = dl_dy[int((i+stride-1)/2), int((j+stride-1)/2), k]
                else:
                    dl_dx[int((i+stride-1)/2), int((j+stride-1)/2), k] = dl_dy[int((i+stride-1)/2), int((j+stride-1)/2), k]
#    print("Pool 2x2 back output dl_dx:", dl_dx.shape)
    return dl_dx


def flattening(x):
    y = x.flatten()
#    print("Flatten y before reshape : ",y.shape)
    y = np.reshape(y,(y.shape[0],1),order='F')
#    print("Flatten y after reshape : ",y.shape)
    return y


def flattening_backward(dl_dy, x, y):
    #dLdx = reshape(dLdy, size(x));
    dl_dx = np.reshape(dl_dy,x.shape,order='F')
#    print("Flatten back output dl/dx : ",dl_dx.shape)
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    #1.Set the learning rate
    learning_rate = 0.01
    #2.Set deacy rate
    decay_rate = 0.5
    #3.Initialize the weights with a Gaussian noise
    w = np.random.normal(0, 1, (10, 196))
    b = np.random.normal(0, 1, (10, 1))
    #4.k=1
    k = 1
    number_of_batches = len(mini_batch_x[1])
    batch_size = np.size(mini_batch_x[1][1])
    num_of_iterations = 3000
    
    #5.Iterate fo the num of iterations
    for iterations in range(1, num_of_iterations):
        #6.At every 1000th iteration, do learning_rate = decay_rate*learning_rate
        if iterations % 1000 == 0:
            learning_rate = decay_rate*learning_rate
        #7.
        dl_dw = 0 
        dl_db = 0
        
        #8.For each image in kth batch 
        for index in range(0, batch_size):
            mini_batch_val_x = mini_batch_x[k]
            x = mini_batch_val_x[:,index]
            x = np.reshape(x, (196,1))
            #9. Predict y
            y_tilde = fc(x, w, b)
            mini_batch_val_y = mini_batch_y[k]
            y = mini_batch_val_y[:, index]
            y = np.reshape(y, (10, 1))
            #10. Compute Loss
            l, dl_dy = loss_euclidean(y_tilde, y)
            #11.Gradient back-propagation
            dl_dx, del_dw, del_db = fc_backward(dl_dy, x, w, b, y_tilde)
            #12.
            dl_dw = dl_dw + del_dw
            dl_db = dl_db + del_db
        #14.Increment k by 1 and check for k value
        k = k + 1
        if k > number_of_batches:
            k = 1
        #15.Update weights and bias
        w = w - (np.dot(learning_rate, dl_dw)/batch_size)
        b = b - (np.dot(learning_rate, dl_db)/batch_size)
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    
    #1.Set the learning rate
    learning_rate = 0.2
    #2.Set deacy rate
    decay_rate = 0.9
    #3.Initialize the weights with a Gaussian noise
    w = np.random.normal(0, 1, (10, 196))
    b = np.random.normal(0, 1, (10, 1))
    #4.k=1
    k = 1
    number_of_batches = len(mini_batch_x[1])
    batch_size = np.size(mini_batch_x[1][1])
    num_of_iterations = 3000
    
    #5.Iterate fo the num of iterations
    for iterations in range(1, num_of_iterations):
        #6.At every 1000th iteration, do learning_rate = decay_rate*learning_rate
        if iterations % 1000 == 0:
            learning_rate = decay_rate*learning_rate
        #7.
        dl_dw = 0 
        dl_db = 0
        
        #8.For each image in kth batch 
        for index in range(0, batch_size):
            mini_batch_val_x = mini_batch_x[k]
            x = mini_batch_val_x[:,index]
            x = np.reshape(x, (196,1))
            #9. Predict y
            y_tilde = fc(x, w, b)
            mini_batch_val_y = mini_batch_y[k]
            y = mini_batch_val_y[:, index]
            y = np.reshape(y, (10, 1))
            #10. Compute Loss <- NOTE --- Change from eucledian to softmax
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            #11.Gradient back-propagation
            dl_dx, del_dw, del_db = fc_backward(dl_dy, x, w, b, y_tilde)
            #12.
            dl_dw = dl_dw + del_dw
            dl_db = dl_db + del_db
        #14.Increment k by 1 and check for k value
        k = k + 1
        if k > number_of_batches:
            k = 1
        #15.Update weights and bias
        w = w - (np.dot(learning_rate, dl_dw)/batch_size)
        b = b - (np.dot(learning_rate, dl_db)/batch_size)
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    w1 = np.random.normal(0, 1, (30, 196))
    b1 = np.random.normal(0, 1, (30, 1))
    w2 = np.random.normal(0, 1, (10, 30))
    b2 = np.random.normal(0, 1, (10, 1))

    learning_rate = 0.283
    decay_rate = 0.97
    k = 1
    number_of_batches = len(mini_batch_x[1])
    batch_size = np.size(mini_batch_x[1][1])
    num_of_iterations = 10000
    
    for iterations in range(1, num_of_iterations):
        #6.At every 1000th iteration, do learning_rate = decay_rate*learning_rate
        if iterations % 1000 == 0:
            learning_rate = decay_rate*learning_rate
        #7.
        dl_dw1 = 0 
        dl_db1 = 0
        dl_dw2 = 0 
        dl_db2 = 0
        
        #8.For each image in kth batch 
        for index in range(0, batch_size):
            mini_batch_val_x = mini_batch_x[k]
            x = mini_batch_val_x[:,index]
            x = np.reshape(x, (196,1))
            #Forward
            a1 = fc(x, w1, b1)
            f1 = relu(a1)
            a2 = fc(f1, w2, b2)
            
            mini_batch_val_y = mini_batch_y[k]
            y = mini_batch_val_y[:, index]
            y = np.reshape(y, (10, 1))
            l, dl_dy = loss_cross_entropy_softmax(a2, y)
            #Bakcward 
            dl_dx2, del_dw2, del_db2 = fc_backward(dl_dy, f1,w2,b2,a2)
            dl_dy1 = relu_backward(dl_dx2, a1, f1)
            dl_dx1, del_dw1, del_db1 = fc_backward(dl_dy1, x, w1, b1, a1)
            #12.
            dl_dw1 = dl_dw1 + del_dw1
            dl_db1 = dl_db1 + del_db1
            dl_dw2 = dl_dw2 + del_dw2
            dl_db2 = dl_db2 + del_db2
        #14.Increment k by 1 and check for k value
        k = k + 1
        if k > number_of_batches:
            k = 1
        #15.Update weights and bias
        w1 = w1 - (np.dot(learning_rate, dl_dw1)/batch_size)
        b1 = b1 - (np.dot(learning_rate, dl_db1)/batch_size)
        w2 = w2 - (np.dot(learning_rate, dl_dw2)/batch_size)
        b2 = b2 - (np.dot(learning_rate, dl_db2)/batch_size)
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    
    w_conv = np.random.normal(0, 1, (3, 3, 1, 3))
    b_conv = np.random.normal(0, 1, (3, 1))
    w_fc = np.random.normal(0, 1, (10, 147))
    b_fc = np.random.normal(0, 1, (10, 1))

    learning_rate = 0.15 #0.2
    decay_rate = 0.9  #0.6
    k = 1
    number_of_batches = len(mini_batch_x[1])
    batch_size = np.size(mini_batch_x[1][1])
    print(batch_size)
    num_of_iterations = 25000

    for iterations in range(1, num_of_iterations):
        print("ITERATION",iterations)
        if iterations % 1000 == 0:
            learning_rate = decay_rate*learning_rate
        dl_dw_conv = 0 
        dl_db_conv = 0
        dl_dw_fc = 0
        dl_db_fc = 0
        for index in range(0, batch_size):
            #Forward
            mini_batch_val_x = mini_batch_x[k]
            x = mini_batch_val_x[:,index]
            x = np.reshape(x, (14, 14, 1),order ='F')
            convoluted_x = conv(x, w_conv, b_conv)
            activated_x = relu(convoluted_x)
            maxpooled_x = pool2x2(activated_x)
            flattened_x = flattening(maxpooled_x)
            fc_x = fc(flattened_x, w_fc, b_fc)

            # Error calculation
            mini_batch_val_y = mini_batch_y[k]
            y = mini_batch_val_y[:, index]
            y = np.reshape(y, (10, 1))
            l, dl_dy = loss_cross_entropy_softmax(fc_x, y)
            
            #Bakcward
            dl_dx, del_dw_fc, del_db_fc = fc_backward(dl_dy, flattened_x, w_fc, b_fc, fc_x)
            dl_dx_flattening = flattening_backward(dl_dx, maxpooled_x, flattened_x)
            dl_dx_pooled = pool2x2_backward(dl_dx_flattening, activated_x, maxpooled_x)
            dl_dx_activation = relu_backward(dl_dx_pooled, convoluted_x, activated_x)
            del_dw_conv, del_db_conv = conv_backward(dl_dx_activation, x, w_conv, b_conv, convoluted_x)

            dl_dw_conv = dl_dw_conv + del_dw_conv
            dl_db_conv = dl_db_conv + del_db_conv
            dl_dw_fc = dl_dw_fc + del_dw_fc
            dl_db_fc = dl_db_fc + del_db_fc
        k = k + 1
        if k > number_of_batches:
            k = 1
        w_conv = w_conv - (np.dot(learning_rate, dl_dw_conv)/batch_size)
        b_conv = b_conv - (np.dot(learning_rate, dl_db_conv)/batch_size)
        w_fc = w_fc - (np.dot(learning_rate, dl_dw_fc)/batch_size)
        b_fc = b_fc - (np.dot(learning_rate, dl_db_fc)/batch_size)
        
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



