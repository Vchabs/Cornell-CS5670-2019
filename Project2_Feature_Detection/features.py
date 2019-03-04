import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def saveHarrisImage(self, harrisImage, srcImage):
        '''
        Saves a visualization of the harrisImage, by overlaying the harris
        response image as red over the srcImage.

        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        '''
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'

        # TODO-BLOCK-BEGIN
        Ix = scipy.ndimage.sobel(srcImage,axis=0)
        Iy = scipy.ndimage.sobel(srcImage,axis=1)

        H = np.zeros((2,2))

        Ix2_weighted = scipy.ndimage.filters.gaussian_filter(Ix*Ix, sigma=0.5)
        Iy2_weighted = scipy.ndimage.filters.gaussian_filter(Iy*Iy, sigma=0.5)
        Ixy_weighted = scipy.ndimage.filters.gaussian_filter(Ix*Iy, sigma=0.5)
        #Loop through each pixel in Ix,Iy/the image
        for y in range(0,height):
            for x in range(0,width):
                # print(img[x-2:x+3,y-2:y+3])
                H[0][0] = Ix2_weighted[y][x]
                H[1][0] = Ixy_weighted[y][x]
                H[0][1] = Ixy_weighted[y][x]
                H[1][1] = Iy2_weighted[y][x]
                c = np.linalg.det(H) - 0.1*np.trace(H)**2
                harrisImage[y][x] = c
                
                orientationImage[y][x] = np.degrees(np.arctan2(Ix[y][x], Iy[y][x]))
        # raise Exception("TODO 1: in features.py not implemented")
        # # TODO-BLOCK-END

        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        maxImage = np.zeros(harrisImage.shape)
        scipy.ndimage.filters.maximum_filter(harrisImage, size=7, output=maxImage, mode='constant', cval=0.0)
        #raise Exception("TODO 2: in features.py not implemented")
        indices = np.where(maxImage == harrisImage)
        row_indices = indices[0]
        col_indices = indices[1]
        for i in range(row_indices.shape[0]):
            if harrisImage[row_indices[i]][col_indices[i]] == maxImage[row_indices[i]][col_indices[i]]:
                destImage[row_indices[i]][col_indices[i]] = True
        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                f.size = 10
                f.pt = (x,y)
                f.angle = orientationImage[y,x]
                f.response = harrisImage[y,x]
                #raise Exception("TODO 3: in features.py not implemented")
                # TODO-BLOCK-END

                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):
            x, y = f.pt
            x, y = int(x), int(y)

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # TODO-BLOCK-BEGIN

            #x-range => [(x-2),(x-1),x,(x+1),(x+2)]
            #y-range => [(y-2),(y-1),y,(y+1),(y+2)]
            desc_index = 0
            for curr_y in xrange(y-2,y+3):
                for curr_x in xrange(x-2,x+3):
                    if curr_y >= 0 and curr_y < image.shape[0]:
                        if curr_x >= 0 and curr_x < image.shape[1]:
                            desc[i][desc_index] = grayImage[curr_y,curr_x]
                    desc_index+=1
            #raise Exception("TODO 4: in features.py not implemented")
            # TODO-BLOCK-END
    
        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN

            # Getting the basic information about the current keypoint's features
            x,y = f.pt
            angle = -1*np.radians(f.angle)
            # We get the translation array we need to do the initial translation 
            # of the 40 by 40 subgrid of the image,. It will be in the form
            # np.array([-x,-y, 0]) according to the slides at: 
            # http://www.cs.cornell.edu/courses/cs4670/2018sp/lec14_descriptors.pdf
            translation1 = transformations.get_trans_mx(np.array([-x,-y,0]))
            # We get the transformation matrix needed to get our 40 by 40 grid 
            # rotated properly.
            rotation = transformations.get_rot_mx(0.0,0.0,angle)
            # We get the transformation matrix needed to scale down the original
            # 40 by 40 grid we obtain by a factor of 5
            scale = transformations.get_scale_mx(0.2,0.2,1)
            # The last transformation is a translation that is half the size of the
            # downsampled window which results in it being np.array([4,4,0]) in our case
            translation2 = transformations.get_trans_mx(np.array([4,4,0]))
            # Finally we use the formula T = M_{T2}M_{S}M_{R}M_{T1} to determine the
            # overall transformation we make to the undelrying image
            overall_transformation = np.dot(np.dot(np.dot(translation2,scale),rotation),translation1)
            # Now we form the correct 2 by 3 numpy array to carry out all of the necessary
            # transformations on the given image by using our calculated overall_transformation
            # and filling in missing entries in the transMx numpy array
            
            # 4 by 4 matrix transforms (x,y,z,1) to (x',y',z',1) 
            # Kai's advice (Piazza) -> 3rd row of 4 by 4 always constant
            # -> the 2D transformation is from (x,y,1) to (x', y')
            # Conclusion: (1) Take out the 3rd column that represents z
            # (2) Take out the last two rows that are always constant
            overall_transformation = np.delete(overall_transformation,2,1)
            overall_transformation = np.delete(overall_transformation,2,0)
            overall_transformation = np.delete(overall_transformation,2,0)

            transMx = overall_transformation
            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            # Subtract out the mean from each of the pixels of the descriptor
            # and then divide by the variance of the pixel values in order to 
            # normalize
            destImageTarget = destImage[:8, :8]
            # numeric issue, set a very small threshold instead of zero
            if np.std(destImageTarget) <= 0.00001:
                desc[i,:] = np.zeros((windowSize * windowSize,))
            else:
                destImageTarget-=np.mean(destImageTarget)
                destImageTarget/=np.std(destImageTarget)
                desc[i,:] = destImageTarget.flatten()
            #raise Exception("TODO 6: image n features.py not implemented")
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        # Let's take advantage of the fact that the feature descriptors
        # have the same number of keypoints
        for i in xrange(desc1.shape[0]):
            ssds = (np.sum((desc2-desc1[i])**2,axis=1))**0.5
            min_ssd_index = np.argmin(ssds)
            match = cv2.DMatch(i, min_ssd_index, ssds[min_ssd_index])
            matches.append(match)
        # TODO-BLOCK-END
        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN
        #raise Exception("TODO 8: in features.py not implemented")
        for i in xrange(desc1.shape[0]):
            ssds = (np.sum((desc2-desc1[i])**2,axis=1))**0.5
            min_ssd_indices = ssds.argsort()[:2]
            ratio_distance = ssds[min_ssd_indices[0]]/ssds[min_ssd_indices[1]]
            match = cv2.DMatch(i, min_ssd_indices[0], ratio_distance)
            matches.append(match)
        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))

