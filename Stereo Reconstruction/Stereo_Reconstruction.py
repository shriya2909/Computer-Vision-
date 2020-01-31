import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

def find_match(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    
    x1_L_R = []
    x2_L_R = []
    nbrs1 = NearestNeighbors(n_neighbors=2).fit(des1)
    distances1, indices1 = nbrs1.kneighbors(des2)
    for i in range(0, distances1.shape[0]):
        if distances1[i,0] < (0.7 * distances1[i,1]):  
            x2_L_R.append([kp2[i].pt[0], kp2[i].pt[1]])
            x1_L_R.append([kp1[indices1[i,0]].pt[0], kp1[indices1[i,0]].pt[1]])
        
    x1_R_L = []
    x2_R_L = []
    nbrs2 = NearestNeighbors(n_neighbors=2).fit(des2)
    distances2, indices2 = nbrs2.kneighbors(des1)
    for i in range(0, distances2.shape[0]):
        if distances2[i,0] < 0.70*distances2[i,1]:
            x1_R_L.append([kp1[i].pt[0], kp1[i].pt[1]])
            x2_R_L.append([kp2[indices2[i,0]].pt[0], kp2[indices2[i,0]].pt[1]])
    
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
            

    pts1 = np.asarray(x1)
    pts2 = np.asarray(x2)
    return pts1, pts2


def compute_F(pts1, pts2):
    ransac_iterations = 1000
    inliers = np.zeros((ransac_iterations, 1))
    F_matrix = []
    ransac_thr = 0.01
    for i in range(ransac_iterations):
        indices = np.random.permutation(pts1.shape[0])
        num_inliers = 0
        A = np.array([ [pts1[indices[0], 0]*pts2[indices[0], 0], pts1[indices[0], 1]*pts2[indices[0], 0], pts2[indices[0], 0], pts1[indices[0], 0]*pts2[indices[0], 1], pts1[indices[0], 1]*pts2[indices[0], 1], pts2[indices[0], 1], pts1[indices[0], 0], pts1[indices[0], 1], 1], 
                       [pts1[indices[1], 0]*pts2[indices[1], 0], pts1[indices[1], 1]*pts2[indices[1], 0], pts2[indices[1], 0], pts1[indices[1], 0]*pts2[indices[1], 1], pts1[indices[1], 1]*pts2[indices[1], 1], pts2[indices[1], 1], pts1[indices[1], 0], pts1[indices[1], 1], 1],
                       [pts1[indices[2], 0]*pts2[indices[2], 0], pts1[indices[2], 1]*pts2[indices[2], 0], pts2[indices[2], 0], pts1[indices[2], 0]*pts2[indices[2], 1], pts1[indices[2], 1]*pts2[indices[2], 1], pts2[indices[2], 1], pts1[indices[2], 0], pts1[indices[2], 1], 1],
                       [pts1[indices[3], 0]*pts2[indices[3], 0], pts1[indices[3], 1]*pts2[indices[3], 0], pts2[indices[3], 0], pts1[indices[3], 0]*pts2[indices[3], 1], pts1[indices[3], 1]*pts2[indices[3], 1], pts2[indices[3], 1], pts1[indices[3], 0], pts1[indices[3], 1], 1],
                       [pts1[indices[4], 0]*pts2[indices[4], 0], pts1[indices[4], 1]*pts2[indices[4], 0], pts2[indices[4], 0], pts1[indices[4], 0]*pts2[indices[4], 1], pts1[indices[4], 1]*pts2[indices[4], 1], pts2[indices[4], 1], pts1[indices[4], 0], pts1[indices[4], 1], 1],
                       [pts1[indices[5], 0]*pts2[indices[5], 0], pts1[indices[5], 1]*pts2[indices[5], 0], pts2[indices[5], 0], pts1[indices[5], 0]*pts2[indices[5], 1], pts1[indices[5], 1]*pts2[indices[5], 1], pts2[indices[5], 1], pts1[indices[5], 0], pts1[indices[5], 1], 1],
                       [pts1[indices[6], 0]*pts2[indices[6], 0], pts1[indices[6], 1]*pts2[indices[6], 0], pts2[indices[6], 0], pts1[indices[6], 0]*pts2[indices[6], 1], pts1[indices[6], 1]*pts2[indices[6], 1], pts2[indices[6], 1], pts1[indices[6], 0], pts1[indices[6], 1], 1],
                       [pts1[indices[7], 0]*pts2[indices[7], 0], pts1[indices[7], 1]*pts2[indices[7], 0], pts2[indices[7], 0], pts1[indices[7], 0]*pts2[indices[7], 1], pts1[indices[7], 1]*pts2[indices[7], 1], pts2[indices[7], 1], pts1[indices[7], 0], pts1[indices[7], 1], 1]
                    ])

        U, S, V = np.linalg.svd(A)
        F_temp = V[-1].reshape(3,3)
        U, D, V = np.linalg.svd(F_temp)
        D = np.array([ [D[0], 0, 0], [0, D[1], 0], [0, 0, D[2]] ])
        D[2,2] = 0
        F_temp = np.dot(U, np.dot(D,V))
        for j in range(pts1.shape[0]):
            V = np.array([pts2[j, 0], pts2[j, 1], 1])
            U = np.array([[pts1[j, 0]], [pts1[j, 1]], [1]])
            V = np.reshape(V, (1, V.shape[0]))
            error = np.absolute(np.dot(V, np.dot(F_temp, U)))
            if error < ransac_thr:
                num_inliers = num_inliers + 1
        
        inliers[i, 0] = num_inliers
        F_matrix.append(F_temp)

    index = np.unravel_index(np.argmax(inliers, axis=None), inliers.shape)
    F = F_matrix[index[0]]
    
    return F


def triangulation(P1, P2, pts1, pts2):
    
    
    
    X = [ [0]*3 for i in range(pts1.shape[0])]
    for i in range(0,pts1.shape[0]):
        A_upper = np.asarray([[0, -1, pts1[i][1]],[1, 0,  -pts1[i][0]],[-pts1[i][1], pts1[i][0], 0]])
        A_upper = np.dot(A_upper,P1)
        A_lower = np.asarray([[0, -1, pts2[i][1]],[1, 0,  -pts2[i][0]],[-pts2[i][1], pts2[i][0], 0]])
        A_lower = np.dot(A_lower,P2)
        A =  np.vstack((A_upper, A_lower)) 
        U,S,V = np.linalg.svd(A)
        temp = V[-1,:4]
        
        pts =np.asarray([temp[0]/temp[3], temp[1]/temp[3], temp[2]/temp[3]]);
        X[i] = pts  
    
    pts3D = X

    return np.asarray(pts3D)


def disambiguate_pose(Rs, Cs, pts3Ds):
    
    
    ##nvalid 4x1 = 0 
    n_valid = np.asarray([0,0,0,0])
   
    ##Loop for item in Rs/Cs : index -> (0,3) : i 
    for i in range(0,4):
        ##extract curr rotation last row
        curr_R_z = np.asarray(Rs[i][2])
        
        
        #extract curr center 
        curr_C = np.asarray(Cs[i])
        
        
        ##Loop for each point in pts3Ds:
        for k in range(pts3Ds[i].shape[0]):
            curr_point = np.reshape(np.asarray(pts3Ds[i][k]),(3,1))
           
            
            #Subtarct curr center from curr point -> diff
            diff = np.subtract(curr_point,curr_C)
            
            
            #Dot product of curr rotation last row and diff -> prod
            prod = np.dot(curr_R_z,diff)
           
            #Check if prod > 0 :
            if prod > 0 :
                #increase :  valid[i] + + 
                n_valid[i] = n_valid[i] + 1
    
    
    #index_max_valid_points  = index for max_valid_points 
    max_ind = np.argmax(n_valid)
    
    #return R,C and pts3D with max_valid_points  using Rs[index_max_valid_points]
    R = Rs[max_ind]
    C = Cs[max_ind]
    pts3D = pts3Ds[max_ind]
    return R, C, pts3D


def compute_rectification(K, R, C):

    rx = C/np.linalg.norm(C)
    
    rz_tilde = np.array([[0], [0], [1]])
    
    dot_rzt_rx_t_trans = np.dot(np.transpose(rz_tilde),rx)*rx
    
    diff = rz_tilde - dot_rzt_rx_t_trans
    diff_norm = np.linalg.norm(diff)
    rz = diff/diff_norm
    
    ry = np.cross(rz,rx, axis = 0)
    
    R_rect = np.reshape(np.array([[rx.T],[ry.T],[rz.T]]),(3,3))
    
    H1 = np.matmul(K,np.matmul(R_rect,np.linalg.inv(K)))
    H2 = np.matmul(K,np.matmul(R_rect,np.matmul(np.transpose(R),np.linalg.inv(K))))
    return H1, H2


def dense_match(img1, img2):

    sift = cv2.xfeatures2d.SIFT_create()
    kp = [cv2.KeyPoint(x, y, 2) for y in range(0, img1.shape[0]) 
                                    for x in range(0, img1.shape[1])]
    
    kp1, dense_feature1 = sift.compute(img1, kp)
    kp2, dense_feature2 = sift.compute(img2, kp)
    
    dense_feature1 = np.reshape(dense_feature1,(img1.shape[0],img1.shape[1],128))
   
    dense_feature2 = np.reshape(dense_feature2,(img2.shape[0],img2.shape[1],128))
    
    disparity=np.zeros((img1.shape[0],img1.shape[1]))
    
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i][j] != 0 :
                temp_df_1 = dense_feature1[i,j,:]
                diff_min = sys.maxsize
                min_ind = -1
                for k in range(0,j):
                    temp_df_2 = dense_feature2[i,k,:]
                    diff = temp_df_2 - temp_df_1
                    p_d_curr = np.linalg.norm(diff)
                    if diff_min > p_d_curr :
                        diff_min = p_d_curr
                        min_ind = k 
                if j != 0 : 
                    index_min = min_ind
                else :
                    index_min = 0 
                
                disparity[i,j] = np.abs(index_min-j)
        
            
   
    
            
    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)
    
    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)
#
#   # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)
#
     #Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)
#
    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)
#
    
    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

#    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
