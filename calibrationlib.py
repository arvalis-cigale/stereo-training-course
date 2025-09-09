"""
"Extraction of 3D information from binocular images"
EPPS Training School 12.09.2025
license under CC By NC 4.0

__author__ = "Samuel Thomas"
__email__ = "s.thomas@arvalis.fr"
__status__ = "Production"
__version__ = "1.0"
"""

import glob as glob
import os

import cv2 as cv
import numpy as np

class CameraCalibration:

    # We define the parameters that will be used for the calibration,
    # especially the characteristics of the chessboard
    def __init__(self, chessboard_pattern, chessboard_cell_size):

        self.MAX_ITER_CRITERIA = 5000
        self.MIN_ERROR_CRITERIA = 1e-16
        self.CHESSBOARD_PATTERN = chessboard_pattern
        self.CHESSBOARD_CELLSIZE = chessboard_cell_size

    # Calibration function for a single camera ; each camera of a stereo pair has
    # to be individually calibrated first
    def single_calibration(
        self,
        image_path,
        image_type,
        log_file=None,
        images_to_skip=None,
    ):

        flags = cv.CALIB_CB_ADAPTIVE_THRESH
        flags |= cv.CALIB_CB_FAST_CHECK
        flags |= cv.CALIB_CB_NORMALIZE_IMAGE
        flags |= cv.CALIB_CB_ACCURACY

        # Output criteria for the calibration function
        criteria = (
            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
            self.MAX_ITER_CRITERIA,
            self.MIN_ERROR_CRITERIA,
        )

        # Preparation of the object points array, i.e the id of each chessboard corner in
        # the (id_x, id_y, id_z=0) coordinate system (id_z=0 as we consider the chessboard plan)
        object_points = np.zeros(
            (self.CHESSBOARD_PATTERN[0] * self.CHESSBOARD_PATTERN[1], 3), np.float32
        )
        # Coordinates are at this point : (0,0,0), (1,0,0), (2,0,0), ..., (0,1,0), ...,(12,7,0)
        object_points[:, :2] = np.mgrid[
            0 : self.CHESSBOARD_PATTERN[0], 0 : self.CHESSBOARD_PATTERN[1]
        ].T.reshape(-1, 2)
        # We multiply the coodinates by the chessboard cell size in mm, so that calibration parameters
        # are expressed in mm rather than in cell counts.
        # For our chessboard, it leads to (0,0,0), (40,0,0), (80,0,0), ..., (0,1,0)
        object_points = object_points * self.CHESSBOARD_CELLSIZE

        # Initialization of the lists used for storing the whole object points and images points for all images.
        # List of 3d points in real world
        object_points_list = []  
        # List of 2d points in the image plan
        image_points_list = []  
        # List of the retained images by the cv.findChessboardCorners function
        success_images_list = (
            []
        )        
        # List of the rejected images by the cv.findChessboardCorners function
        missed_images_list = (
            []
        )

        # Initialization of the needed variables
        camera_matrix = []        
        roi = []
        dist_coefs = []
        r_vecs = []
        t_vecs = []
        gray = np.zeros((1, 1))

        ret = False

        # List of the chessboard images to be processed
        images_list = glob.glob(image_path + "/" + f"*Camera*.{image_type}")
        images_list.sort()

        if len(images_list) > 0:

            log_line = str(len(images_list)) + " images found\r"
            print(log_line)

            log_file.write(log_line)

            # Loop on the images of the list
            cnt = 0
            for img_name in images_list:

                print("Entering image " + img_name)

                log_line = "Entering image " + img_name + "\r"

                if log_file != None:
                    log_file.write(log_line)

                img_id = cnt

                # Our images are named this way : "SOMETHING_CameraX_Y.ext", where :
                # X is the id of the camera (1: left - 2: right)
                # Y i the if of the image for the CameraX 
                # /!\ in this code we assume that for a same pair of images, 
                # Y must be the same value for Camera1 and Camera2)
                # ex : AAA_Camera1_5.jpg and AAA_Camera2_5.jpg are respectively
                # the left and right images of the same point of view at the same time

                # We plit the iamge name to retrieve that 'Y' image id
                if img_name.find("Camera") > -1:
                    img_id = img_name.split("Camera")
                    img_id = img_id[1].split(".")
                    img_id = img_id[0].split("_")
                    img_id = img_id[1]

                # We load the current image
                img = cv.imread(img_name)
                h, w = img.shape[:2]

                # We convert it into a grayscale image
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # We launch the search function for the corners of the chessboard
                ret, corners = cv.findChessboardCorners(
                    gray, self.CHESSBOARD_PATTERN, flags=flags
                )

                # We create a 'path_calibration / "Left_corners"' 
                # or 'path_calibration / "Right_corners"' directory
                # (depending on the current camera being calibrated)
                # so that we can save the chessboard images with 
                # output corners detection, when the function succeeds
                camera_id = image_path.split("/")
                camera_id = camera_id[len(camera_id) - 1] + "_corners"
                index = image_path.rfind("/")
                out_name = image_path[:index] + "/" + camera_id

                # We make the directory if it doesn't exist yet
                if not os.path.exists(out_name):
                    os.makedirs(out_name)

                # If the corner search is successful, the object points are added to the corresponding list, 
                # and image points will be too, after refining the search around the image points found above.
                if ret == True:

                    success_images_list.append(img_id)

                    # Addition of object points to the list of those already detected in previous images
                    object_points_list.append(object_points)

                    criteria = (
                        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                        self.MAX_ITER_CRITERIA * 2,
                        self.MIN_ERROR_CRITERIA / 10,
                    )

                    # We refine the coordinates of the corners in the image plane, then addition of the image 
                    # points to the list of those already detected in the previous images
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    image_points_list.append(corners2)

                    # We draw refined corners on the chessboard input image and save it in the 
                    # 'path_calibration / "Left_corners"' or 'path_calibration / "Right_corners"'
                    # directory, depending on the current camera being calibrated
                    draw_corners = cv.drawChessboardCorners(
                        img, self.CHESSBOARD_PATTERN, corners2, ret
                    )

                    out_name = out_name + "/Corner_" + img_id + ".jpg"
                    cv.imwrite(out_name, draw_corners)

                    if log_file != None:
                        log_file.write(log_line)

                else:
                    # Si la recherche n'a pas abouti, on ajoute l'id de l'image courante (i.e.
                    # juste l'index de l'image dans la série, sans le 'CameraX_' ,pour matcher
                    # facilement les images issues de la première caméra avce celles de la seconde).
                    # à la liste des images non prises en compte / à ne pas prendre en compte
                    # pour la suite
                    # If the search was unsuccessful, add the id of the current image (i.e.
                    # just the index of the image in the series, without “CameraX_”, to easily match
                    # images from the first camera with those from the second) to the list of images
                    # not taken into account / not to be taken into account for the stereo calibration                    
                    missed_images_list.append(img_id)

                    if not os.path.exists(out_name):
                        os.makedirs(out_name)

                    out_name = out_name + "/Rejected_" + img_id + ".jpg"
                    cv.imwrite(out_name, img)

                    print("Rejected")

                    log_line = "Rejected\r"

                    if log_file != None:
                        log_file.write(log_line)

                cnt += 1

            # End of the loop on the chessboard images ; we are now ready to launch the camera calibration
            # that will lead to intrinsic and extrinsic camera parameters            

            # We choose reasonable and appropriate flags and parameters, from the description given in the
            # OpenCV documentation (see "./documentation/OpenCV - calibrateCamera.pdf")
            flags = cv.CALIB_TILTED_MODEL

            criteria = (
                cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                self.MAX_ITER_CRITERIA,
                self.MIN_ERROR_CRITERIA,
            )

            # We launch the calibration of the camera
            ret, camera_matrix, dist_coefs, r_vecs, t_vecs = cv.calibrateCamera(
                object_points_list, image_points_list, (w, h), flags, criteria
            )

            # We compute the error for each object point, and add it to mean_error, so that 
            # we get a mean reprojection error at the end
            mean_error = 0
            for i in range(len(object_points_list)):
                img_pre, _ = cv.projectPoints(
                    object_points_list[i], r_vecs[i], t_vecs[i], camera_matrix, dist_coefs
                )
                error = cv.norm(image_points_list[i], img_pre, cv.NORM_L2) / len(
                    img_pre
                )
                mean_error += error

            # We commpute the mean reprojection error on the whole considered object points 
            reprojection_error = mean_error / len(object_points_list)
            
            print("Mean error = ", mean_error)
            print("Headcount = ", len(object_points_list))
            print("Reprojection error = ", reprojection_error)
            print("Rejected images : ", missed_images_list)
            log_line = (
                "Mean error = "
                + str(mean_error)
                + "\rHeadcount = "
                + str(len(object_points_list))
                + "\rReprojection error = "
                + str(reprojection_error)
                + "\rRejected images : "
                + str(missed_images_list)
                + "\r"
            )

            if log_file != None:
                log_file.write(log_line)

            ret = True

        else:
            print("No image found here  : ", image_path)

            ret = False

        return (
            ret,
            reprojection_error,
            success_images_list,
            roi,
            camera_matrix,
            dist_coefs,
            r_vecs,
            t_vecs,
            object_points_list,
            image_points_list,
            gray.shape[::-1],
        )

    # Calibration function for a stereo camera ; each camera of the stereo pair 
    # must have been individually calibrated first
    def stereo_calibration(
        self,
        object_points,
        image_points_cam1,
        image_points_cam2,
        camera_matrix_cam1,
        dist_coefs_cam1,
        camera_matrix_cam2,
        dist_coefs_cam2,
        image_size,
        log_file=None,
    ):

        # We choose reasonable and appropriate flags and parameters, from the description given in the
        # OpenCV documentation (see "./documentation/OpenCV - stereoCalibrate.pdf")
        criteria = (
            cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,
            self.MAX_ITER_CRITERIA,
            self.MIN_ERROR_CRITERIA,
        )

        flags = cv.CALIB_USE_INTRINSIC_GUESS
        flags |= cv.CALIB_USE_EXTRINSIC_GUESS

        # We launch the calibration of the stereo camera
        (
            ret,
            camera_matrix_cam1_stereo,
            dist_coefs_cam1_stereo,
            camera_matrix_cam2_stereo,
            dist_coefs_cam2_stereo,
            R,
            T,
            E,
            F,
        ) = cv.stereoCalibrate(
            object_points,
            image_points_cam1,
            image_points_cam2,
            camera_matrix_cam1,
            dist_coefs_cam1,
            camera_matrix_cam2,
            dist_coefs_cam2,
            image_size,
            flags,
            criteria,
        )

        print("Reprojection error of the stereo pair = ", str(ret))
        
        return (
            ret,
            camera_matrix_cam1_stereo,
            dist_coefs_cam1_stereo,
            camera_matrix_cam2_stereo,
            dist_coefs_cam2_stereo,
            R,
            T,
        )