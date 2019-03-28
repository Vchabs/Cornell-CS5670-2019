import math
import random

import cv2
import numpy as np
from scipy import spatial

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN
        #raise Exception("TODO in alignment.py not implemented")

        # Reference: http://6.869.csail.mit.edu/fa17/lecture/lecture14sift_homography.pdf
        # Slide #9
        A[2*i, 0] = a_x
        A[2*i, 1] = a_y
        A[2*i, 2] = 1
        A[2*i, 6] = -b_x * a_x
        A[2*i, 7] = -b_x * a_y
        A[2*i, 8] = -b_x

        A[2*i+1,3] = a_x
        A[2*i+1,4] = a_y
        A[2*i+1,5] = 1
        A[2*i+1,6] = -b_y * a_x
        A[2*i+1,7] = -b_y * a_y
        A[2*i+1,8] = -b_y
        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    '''print "U = ", U
    print 
    print "s = ", s
    print 
    print "Vt = ", Vt
    print'''

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN
    #raise Exception("TODO in alignment.py not implemented")
    H = Vt[8].reshape(3,3)
    H/=H[2,2]
    #TODO-BLOCK-END
    #END TODO
    print "H = ", H
    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslation) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.
    #TODO-BLOCK-BEGIN
    #raise Exception("TODO in alignment.py not implemented")
    #max_inlier_motion_est = None
    max_inliers = []
    for i in range(nRANSAC):
        if m == eHomography: # Choose 4 random matches
            random_matches = np.random.choice(matches,4)
            current_motion = computeHomography(f1,f2,random_matches)
        else: # Choose only 1 random match
            random_matches = [random.choice(matches)]
            (a_x, a_y) = f1[random_matches[0].queryIdx].pt
            (b_x, b_y) = f2[random_matches[0].trainIdx].pt
            current_motion = np.eye(3)
            current_motion[0,2] = b_x-a_x
            current_motion[1,2] = b_y-a_y
        
        curr_inliers = getInliers(f1,f2,matches,current_motion,RANSACthresh)
        if len(curr_inliers) > len(max_inliers):
            max_inliers = curr_inliers
            #max_inlier_motion_est = current_motion

    M = leastSquaresFit(f1,f2,matches,m,max_inliers)
    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN
        #raise Exception("TODO in alignment.py not implemented")
        current_match = matches[i]
        current_feature1_point = f1[current_match.queryIdx].pt
        current_feature2_point = f2[current_match.trainIdx].pt
        current_feature1_vec = np.zeros(3)
        #current_feature1_vec = np.array([[current_feature1_point[0]],[current_feature1_point[1]],[1]])
        #current_feature2_vec = np.array([[current_feature2_point[0]],[current_feature2_point[1]],[1]])
        current_feature1_vec[0] = current_feature1_point[0]
        current_feature1_vec[1] = current_feature1_point[1]
        current_feature1_vec[2] = 1
        transformed_vec = np.dot(M,current_feature1_vec)
        # Normalize all the coordinates of the transformed vector
        transformed_vec[0]/=transformed_vec[2]
        transformed_vec[1]/=transformed_vec[2]
        #transformed_vec[2,0] = 1.0
        distance_x = (current_feature2_point[0]-transformed_vec[0])**2
        distance_y = (current_feature2_point[1]-transformed_vec[1])**2
        distance = np.sqrt(distance_x+distance_y)
        #distance = spatial.distance.euclidean(current_feature2_vec, transformed_vec)
        print distance, RANSACthresh
        if distance <= RANSACthresh:
            inlier_indices.append(i)
        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN
            #raise Exception("TODO in alignment.py not implemented")
            inlier_match = matches[inlier_indices[i]]
            current_feature1_point = f1[inlier_match.queryIdx].pt
            current_feature2_point = f2[inlier_match.trainIdx].pt
            u += current_feature2_point[0]-current_feature1_point[0]
            v += current_feature2_point[1]-current_feature1_point[1]
            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN
        #raise Exception("TODO in alignment.py not implemented")
        inlier_matches = []
        print len(inlier_indices)
        for i in range(len(inlier_indices)):
            inlier_match = matches[inlier_indices[i]]
            inlier_matches.append(inlier_match)
        M = computeHomography(f1, f2, inlier_matches)
        #print "Current Calculated M = ", M
        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

