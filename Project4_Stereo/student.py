# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """

    albedo = np.zeros(images[0].shape)
    normals = np.zeros((images[0].shape[0], images[0].shape[1], 3))

    num_channels = albedo.shape[2]

    for y in xrange(images[0].shape[0]):
        print str(y) + " out of " + str(images[0].shape[0])
        for x in xrange(images[0].shape[1]):
            for i in range(num_channels):
                intensities = []
                for image in images:
                    intensities.append(image[y,x,i].T)
                intensities = np.array(intensities)
                LTL_inv = np.linalg.inv(lights.T.dot(lights))
                LTI = lights.T.dot(intensities)
                G = LTL_inv.dot(LTI)
                k_d = np.linalg.norm(G)
                print "k_d: " , k_d.shape
                print "G: ", G.shape
                #print G
                if (k_d > 1e-7).all():
                    normals[y,x]+= (G/(k_d*num_channels))
                #print N
                #print
                albedo[y,x,i] = k_d
    return albedo, normals

    '''albedo = np.zeros(images[0].shape)
    normals = np.zeros((images[0].shape[0], images[0].shape[1], 3))

    num_channels = albedo.shape[2]

    for y in xrange(images[0].shape[0]):
        print str(y) + " out of " + str(images[0].shape[0])
        for x in xrange(images[0].shape[1]):
            intensities = np.array([[image[x,y,i] for image in images] for i in range(num_channels)]).T
            LTL_inv = np.linalg.inv(lights.T.dot(lights))
            LTI = lights.T.dot(intensities)
            G = LTL_inv.dot(LTI)
            print "G.shape: ", G.shape
            k_d = np.linalg.norm(G, axis=0)
            print "k_d.shape: ", k_d.shape
            for i in range(num_channels):
                if k_d[i] > 1e-7:
                    normals[y,x] += np.sum(G,axis=1)[i]/(k_d[i]*num_channels)
            albedo[y,x] = k_d
    print "normals = ", normals

    return albedo, normals'''


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    print K.shape, Rt.shape, points.shape
    projection_matrix = K.dot(Rt)
    projections = np.zeros((points.shape[0], points.shape[1], 2))
    for h in range(points.shape[0]):
        for w in range(points.shape[1]):
            actual_point = np.array(points[h,w])
            world_point_4d = np.array([actual_point[0], actual_point[1], actual_point[2], 1.0])
            image_point = projection_matrix.dot(world_point_4d)
            z = image_point[2]
            projections[h,w] = np.array([image_point[0]/z, image_point[1]/z])
    return projections



def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x112, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    height = image.shape[0]
    width = image.shape[1]
    num_channels = image.shape[2]
    normalized = np.zeros((height, width, num_channels*ncc_size*ncc_size))

    print ncc_size, image.shape

    for row in range(ncc_size/2,height-ncc_size/2):
        for col in range(ncc_size/2,width-ncc_size/2):
            patch = image[row-ncc_size/2:row+ncc_size/2+1,col-ncc_size/2:col+ncc_size/2+1,:]
            #print patch.shape
            # (1) Compute the means per channel
            channel_means = np.mean(np.mean(patch, axis=0),axis=0)
            patch_vector=patch-channel_means

            # (2) Normalize the vector
            v = []
            for channel in range(num_channels):
                v.extend(patch_vector[:,:,channel].flatten())
            v_norm = np.array(v)

            if np.linalg.norm(v) >= 1e-6:
                v_norm/=np.linalg.norm(v_norm)
                normalized[row,col] = v_norm

    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    height,width,vector_length = image1.shape
    ncc = np.zeros((height, width))
    for row in range(height):
        for col in range(width):
            ncc[row,col] = np.correlate(image1[row,col], image2[row,col])
    return ncc 

