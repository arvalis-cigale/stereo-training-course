import numpy as np
import cv2 as cv
import laspy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter1d

# Function that loads the camera calibration parameters generated with the calibration code (see calibration.ipynb)
def load_calibration_from_numpy(calibration_path):

    img_size_left = None
    camera_mtx_left = None
    dist_coefs_left = None
    img_size_right = None
    camera_mtx_right = None
    dist_coefs_right = None
    R = None
    T = None

    print("Left camera")
    # Intrinsic Matrix for camera 1
    camera_mtx_left = np.load(calibration_path / "Left" / f"camera_matrix_from_stereo.npy")
    # Distorsion coefficients for camera 1
    dist_coefs_left = np.load(calibration_path / "Left" / f"dist_coefs_from_stereo.npy")
    # Size of sensor of camera 1
    img_size_left = np.load(calibration_path / "Left" / f"image_size.npy")
    print("Mean reprojection error :", np.load(calibration_path / "Left" / f"reprojection_error.npy"), "\n")
    
    print("Right camera")
    # Intrinsic Matrix for camera 2
    camera_mtx_right = np.load(calibration_path / "Right" / f"camera_matrix_from_stereo.npy")
    # Distorsion coefficients for camera 2
    dist_coefs_right = np.load(calibration_path / "Right" / f"dist_coefs_from_stereo.npy")
    # Size of sensor of camera 2
    img_size_right = np.load(calibration_path / "Right" / f"image_size.npy")
    print("Mean reprojection error :", np.load(calibration_path / "Right" / f"reprojection_error.npy"), "\n")

    print("Stereo RGB pair")
    # Load rotation and translation matrices resulting of the stereo pair calibration
    # Rotation matrix
    R_vec = np.load(calibration_path / f"R.npy")
    R, _ = cv.Rodrigues(R_vec)
    # Translation matrix
    T = np.load(calibration_path / f"T.npy")
    print("Mean reprojection error :", np.load(calibration_path / f"reprojection_error.npy"), "\n")
    
    return (
        img_size_left,
        camera_mtx_left,
        dist_coefs_left,
        img_size_right,
        camera_mtx_right,
        dist_coefs_right,
        R,
        T,
    )

# Function that loads the camera calibration from yaml for Literal v2
def load_calibration_from_yaml(calibration_filename):

    img_size_left = None
    camera_mtx_left = None
    dist_coefs_left = None
    img_size_right = None
    camera_mtx_right = None
    dist_coefs_right = None
    R = None
    T = None
    serial_number = None
    calibration_datetime = None

    # Read file content
    s = cv.FileStorage()
    s.open(str(calibration_filename), cv.FileStorage_READ)

    ## [Serial number]
    n = s.getNode("Serial number")
    serial_number = n.real()
    print("Serial number :", int(serial_number))

    ## [Calibration date]
    n = s.getNode("Calibration date")
    calibration_datetime = n.string()
    print("Calibration date :", calibration_datetime, "\n")

    ## [camL]
    n = s.getNode("camL")
    print("Left camera")
    camera_mtx_left = n.getNode("cameraMatrix").mat()
    img_size_left = (
        int(n.getNode("imgSize").at(0).real()),
        int(n.getNode("imgSize").at(1).real()),
    )
    print("Distortion model :", n.getNode("distortionModel").string())
    dist_coefs_left = n.getNode("distCoeffs").mat()
    print("Mean reprojection error :", n.getNode("meanReprojErr").real(), "\n")
    ## [camL]

    ## [camR]
    n = s.getNode("camR")
    print("Right camera")
    camera_mtx_right = n.getNode("cameraMatrix").mat()
    img_size_right = (
        int(n.getNode("imgSize").at(0).real()),
        int(n.getNode("imgSize").at(1).real()),
    )
    print("Distortion model :", n.getNode("distortionModel").string())
    dist_coefs_right = n.getNode("distCoeffs").mat()
    print("Mean reprojection error :", n.getNode("meanReprojErr").real(), "\n")
    ## [camR]

    ## [StereoPairL_R]
    n = s.getNode("StereoPairL_R")
    print("Stereo RGB pair")
    R = n.getNode("R").mat()
    R, _ = cv.Rodrigues(R)

    T = n.getNode("T").mat()
    print("Mean reprojection error :", n.getNode("meanReprojErr").real(), "\n")
    ## [StereoPairL_R]

    return (
        img_size_left,
        camera_mtx_left,
        dist_coefs_left,
        img_size_right,
        camera_mtx_right,
        dist_coefs_right,
        R,
        T,
        serial_number,
        calibration_datetime,
    )
    
# Reads an image

def read_image(image_filename, flag=cv.IMREAD_UNCHANGED):

    img = None

    try:
        img = cv.imread(image_filename, flag)
    except:
        print(f"A problem occured reading this image")                
    
    return img
    
# Writes an image
def write_image(image_filename, image):
    
    try:
        cv.imwrite(image_filename, image) 
    except:
        print(f"A problem occured writing this image")
        
# Displays a list of images horizontally
def display_images(images_list):
    
    # Create a subdivision to display images side by side on one row
    fig, axes = plt.subplots(nrows=1, ncols=len(images_list))
    
    if len(images_list) > 1:
        axes = axes.ravel()
                
        for imp, ax in zip(images_list, axes):
            img = mpimg.imread(imp)
            if len(img.shape) == 2:
                cmap='gray'
            else:
                cmap=None
            ax.imshow(img, cmap=cmap)
            ax.axis('off')
        fig.tight_layout()
    else:
        img = mpimg.imread(images_list[0])        
        if len(img.shape) == 2:
            cmap='gray'
        else:
            cmap=None
        axes.axis('off')
        axes.imshow(img, cmap=cmap)
        
# Rectifies an image
def rectify(src_image, map_x_1, map_x_2):

    img = None

    try:        
        dst_img = cv.remap(src_image, map_x_1, map_x_2, cv.INTER_LINEAR)
    except:
        print(f"A problem occured rectifying this image")
    
    return dst_img

# Normalizes an image as a 8 bits image (values in the [0, 255] interval)
def normalize_image(img):
    # Min and max bounds of the input image
    min_src = int(np.min(img))
    max_src = int(np.round(np.max(img)))

    # Max value for 8 bits output : 256 - 1
    max_dst = 2**8 - 1
    
    # Copy of the input image, to ensure it won't be affected
    img_norm = np.copy(img)
    # Normalization of the copied image from [min_src; max_src] to [0; 1] 
    img_norm = (img_norm - min_src) / (max_src - min_src)
    # Multiplication by max_dst (= 255) so that output values are in the [0, 255] interval
    img_norm = max_dst * img_norm

    # Ensure that there are no rounded values outside the wanted interval
    img_norm[img_norm < 0] = 0
    img_norm[img_norm > max_dst] = max_dst

    # Formats the output as a 8 bits unsigned integer image
    img_norm_int = np.copy(np.round(img_norm)).astype(np.uint8)
        
    return img_norm_int

# Writes a point cloud with RGB information into the laz format (compressed 'las')
def write_point_cloud(file_name, xyz_image, bgr_image):

    xVector = xyz_image[:, 0]
    yVector = xyz_image[:, 1]
    zVector = xyz_image[:, 2]

    bVector = bgr_image[:, 0]
    gVector = bgr_image[:, 1]
    rVector = bgr_image[:, 2]

    # Creation of the appropriate las file
    ptCloud = laspy.create(file_version="1.4", point_format=2)
    
    ptCloud.header.offset = [0, 0, 0]
    # Values are given in mm
    ptCloud.header.scale = [0.001, 0.001, 0.001]
    # Nb of points of the point cloud
    ptCloud.header.point_count = len(xVector)

    # Assignment of matching values vector to each layer of the point cloud
    ptCloud.x = xVector
    ptCloud.y = yVector
    ptCloud.z = zVector
    ptCloud.red = rVector
    ptCloud.blue = bVector
    ptCloud.green = gVector

    # Writes the file on the disk
    ptCloud.write(file_name)

# Converts depth value (in mm) to disparity, given the focal length and the baseline of the stereo pair
def compute_disparity_from_depth(baseline, focal_length, depth):

    disparity = -1
    if depth > 0:
        disparity = (baseline * focal_length / depth).astype(np.float32)

    return disparity

# Converts disparity value to depth (in mm), given the focal length and the baseline of the stereo pair
def compute_depth_from_disparity(baseline, focal_length, disparity):

    depth = -1
    if disparity > 0:
        depth = (baseline * focal_length / disparity).astype(np.float32)

    return depth

# Computes the number of disparities needed by the SGBM algorithm from min and max desired values.
# Min value will be kept, but the max disparity value will be potentially increased, as the number 
# of expected disparities must be a multiple of 16 (as implemented in OpenCV, read from the doc). 
def compute_number_of_disparities(min_disparity, max_disparity):

    num_disparities = max_disparity - min_disparity + 1
    num_disparities = num_disparities / 16
    num_disparities = int(num_disparities + 1) * 16

    return num_disparities

# Computes the correspondence matching between each pixel of the left and right image of a stereo camera pair,
# and returns the corresponding disparity map.
def stereo_processing(left_img,
                      right_img,
                      Q,
                      init_min_disparity,
                      init_num_disparities,
                      block_size,
                      uniqueness_ratio,
                      wls_activation=False,
                      wls_sigma=1.2):

    # Return value : ret = 0 if success, < 0 otherwise
    ret = 0

    # Dimensions (height, width) of the input images
    h, w = left_img.shape[:2]

    # Initialization of the output disparity map
    disparity = np.zeros(left_img.shape)

    min_disparity = init_min_disparity
    max_disparity = init_min_disparity + init_num_disparities - 1
    num_disparities = init_num_disparities

    # Creates a matcher object with appropriate parameters 
    # (for our application and from our experience)
    left_matcher = cv.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=64 * block_size**2,
        disp12MaxDiff=25,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=0,
        speckleRange=0,        
        mode=cv.StereoSGBM_MODE_HH4,
    )

    # If active, creates a WLS filter object 
    if wls_activation:
        wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
        lmbda = 8000.0
        sigma = wls_sigma
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

    # Launch the SGBM matching algorithm
    try:
        raw_disparity = left_matcher.compute(
            np.copy(left_img), np.copy(right_img)
        ).astype(np.float32)

        # Rescale disparity output, as explained in the OpenCV documentation
        disparity = np.copy(raw_disparity)
        disparity = disparity / 16

    except ValueError:
        print(str(ValueError))
        ret = -1
    except cv.error as err:
        print(err)
        ret = -2

    if wls_activation:
        
        # Compute disparity from right to left image if WLS filtering is active
        disparity_right = np.zeros(right_img.shape)
        
        try:
            # Creates a matcher object for the right image
            right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

            # Launch the SGBM matching algorithm from the right image
            raw_disparity_right = right_matcher.compute(
                np.copy(right_img), np.copy(left_img)
            ).astype(np.float32)

            # Rescale disparity output, as explained in the OpenCV documentation
            disparity_right = np.copy(raw_disparity_right)
            disparity_right = disparity_right / 16

            # Computes the WLS filtering comparing left to right and right to left 
            # obtained disparity maps (before rescaling)
            disparity = wls_filter.filter(
                np.copy(raw_disparity),
                np.copy(left_img),
                None,
                np.copy(raw_disparity_right),
            )

            disparity[disparity < 0] = 0
            disparity[np.isnan(disparity)] = 0
            disparity[np.isinf(disparity)] = 0

            # Rescale disparity output, as explained in the OpenCV documentation            
            disparity = disparity / 16

            # Generation of the confidence map to get metrics about matching and filtering quality
            confidence_map = wls_filter.getConfidenceMap()

        except ValueError:
            print(str(ValueError))
            ret = -3
        except cv.error as err:
            print(err)
            ret = -4

    return ret, disparity, min_disparity, max_disparity

# Computes the normalized total gradient between the left image, 
# and the right-on-left mapped image obtained by applying disparity
# to each pixel
# Arguments :
#   - Left rectified grayscale image
#   - Right rectified grayscale image
#   - Disparity map (floating values [min_disparity:max_disparity])
def normalized_total_gradient(
    gray_img_left,
    disparity_left,
    gray_img_right,
    min_disparity,
    max_disparity,
):

    # Initialization of the image obtained by mapping the right image on the left one 
    # using disparity value found for each pixel
    right_on_left_img = np.zeros(gray_img_left.shape)
    left_img = np.copy(gray_img_left)

    # Fill the right_on_left_img pixels, taking pixels in the right image at the same line 
    # as in the left one, and at the column index j_right = j_left - disparity[i_left, j_left] 
    # (where valid values uf disparity[i_left, j_left] has been found (i.e. >= min_disparity))
    for i_left in range(disparity_left.shape[0]):
        i_right = i_left
        for j_left in range(disparity_left.shape[1]):
            j_right = j_left - int(disparity_left[i_left, j_left])
            # First condition for a valid j_right index : it must be positive
            if j_right >= 0:
                # We check that the disparity at these indices is valid 
                # (min_disparity-1 = NODATA)
                if (int(disparity_left[i_left, j_left])) >= min_disparity:
                    # If so, fill the right-on-left image pixel at [i_left, j_left]
                    # with the pixel in the right image at [i_right, j_right]
                    right_on_left_img[i_left, j_left] = gray_img_right[i_right, j_right]
                else:
                    # Fill both left and right-on-left pixel with extem values, so that the
                    # resulting gradient is maximal (i.e. not good)
                    right_on_left_img[i_left, j_left] = 0
                    left_img[i_left, j_left] = 255

    right_on_left_img = np.uint8(right_on_left_img)

    # We truncate a band of max_diparity pixels on both left_img and right_on_left_img,
    # because these pixels are not seen in the right image. There is thus no matching 
    # possible in this band
    left_img = left_img[:, max_disparity:]
    right_on_left_img = right_on_left_img[:, max_disparity:]
    
    left_minus_right = cv.absdiff(left_img, right_on_left_img)

    # [Gx1, Gy1] = imgradientxy(I1);
    # [Gx2, Gy2] = imgradientxy(I2);
    # [Gx12, Gy12] = imgradientxy(I1 - I2);
    # ntg = (nanmean(abs(Gx12), [1 2]) + nanmean(abs(Gy12), [1 2])). / (
    #             nanmean(abs(Gx1), [1 2]) + nanmean(abs(Gx2), [1 2]) + nanmean(abs(Gy1), [1 2]) + nanmean(abs(Gy2),
    #                                                                                                        [1 2]));

    grad_left = np.gradient(left_img)
    grad_right = np.gradient(right_on_left_img)
    grad_mix = np.gradient(left_minus_right)

    ntg = (
        np.nanmean(np.abs(grad_mix[0]), (0, 1))
        + np.nanmean(np.abs(grad_mix[1]), (0, 1))
    ) / (
        np.nanmean(np.abs(grad_left[0]), (0, 1))
        + np.nanmean(np.abs(grad_right[0]), (0, 1))
        + np.nanmean(np.abs(grad_left[1]), (0, 1))
        + np.nanmean(np.abs(grad_right[1]), (0, 1))
    )

    return np.mean(ntg), left_minus_right, right_on_left_img

