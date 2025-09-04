"""
Calibration class for a single camera and a stereo pair
juillet 2020
"""

__author__ = "Samuel Thomas"
__email__ = "s.thomas@arvalis.fr"
__status__ = "Production"
__version__ = "1.0"


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

    # Calibration function of a single camera
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

            cnt = 0
            for img_name in images_list:

                print("Entering image " + img_name)

                log_line = "Entering image " + img_name + "\r"

                if log_file != None:
                    log_file.write(log_line)

                img_id = cnt

                if img_name.find("Camera") > -1:
                    img_id = img_name.split("Camera")
                    img_id = img_id[1].split(".")
                    img_id = img_id[0].split("_")
                    img_id = img_id[1]

                # Chargement des images gauche et droite
                img = cv.imread(img_name)
                h, w = img.shape[:2]

                # Conversion BW classique
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                ret, corners = cv.findChessboardCorners(
                    gray, self.CHESSBOARD_PATTERN, flags=flags
                )

                camera_id = image_path.split("/")
                camera_id = camera_id[len(camera_id) - 1] + "_corners"
                index = image_path.rfind("/")
                out_name = image_path[:index] + "/" + camera_id

                # Si le résultat de la recherche des coins est un succès, on ajoute les points objets, puis les points
                # images après affinage de la recherche autour des points trouvés ci-dessus
                if ret == True:

                    success_images_list.append(img_id)

                    # Ajout des points objets à la liste de ceux qui ont déjà été détectés sur les images précédentes
                    object_points_list.append(object_points)

                    criteria = (
                        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                        self.MAX_ITER_CRITERIA * 2,
                        self.MIN_ERROR_CRITERIA / 10,
                    )

                    # Affinage des coordonnées des coins dans le plan image, puis ajout des points images à la liste
                    # de ceux qui ont déjà été détectés sur les images précédentes
                    corners2 = cv.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    # corners2 = cv.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)
                    # corners2 = cv.cornerSubPix(gray, corners, (25, 25), (-1, -1), criteria)
                    # corners2 = cv.cornerSubPix(gray, corners, (50, 50), (-1, -1), criteria)
                    # corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

                    image_points_list.append(corners2)

                    draw_corners = cv.drawChessboardCorners(
                        img, self.CHESSBOARD_PATTERN, corners2, ret
                    )
                    """
                    image_points_list.append(corners)
                    draw_corners = cv.drawChessboardCorners(img, self.CHESSBOARD_PATTERN, corners, ret)
                    """

                    # print('out_name :', out_name)
                    if not os.path.exists(out_name):
                        os.makedirs(out_name)

                    out_name = out_name + "/Corner_" + img_id + ".jpg"
                    cv.imwrite(out_name, draw_corners)

                    if log_file != None:
                        log_file.write(log_line)

                    # Tracé et affichage des coins de l'échiquier
                    # if (cnt > 30):
                    #     img = cv.drawChessboardCorners(img, self.CHESSBOARD_PATTERN, corners2, ret)
                    #     cv.namedWindow('img' + str(cnt), cv.WINDOW_NORMAL)
                    #     cv.imshow('img' + str(cnt), img)
                    #     cv.waitKey(500)
                else:
                    # Si la recherche n'a pas abouti, on ajoute l'id de l'image courante (i.e.
                    # juste l'index de l'image dans la série, sans le 'CameraX_' ,pour matcher
                    # facilement les images issues de la première caméra avce celles de la seconde).
                    # à la liste des images non prises en compte / à ne pas prendre en compte
                    # pour la suite
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

            flags = cv.CALIB_TILTED_MODEL

            criteria = (
                cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                self.MAX_ITER_CRITERIA,
                self.MIN_ERROR_CRITERIA,
            )

            ret, camera_matrix, dist_coefs, r_vecs, t_vecs = cv.calibrateCamera(
                object_points_list, image_points_list, (w, h), flags, criteria
            )

            cv.destroyAllWindows()

            mean_error = 0
            for i in range(len(object_points_list)):
                img_pre, _ = cv.projectPoints(
                    object_points_list[i], r_vecs[i], t_vecs[i], camera_matrix, dist_coefs
                )
                error = cv.norm(image_points_list[i], img_pre, cv.NORM_L2) / len(
                    img_pre
                )
                mean_error += error

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

        # Calibration du système stéréo : affinage des paramètres intrinsèques et de distorsion de chaque caméra
        criteria = (
            cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,
            self.MAX_ITER_CRITERIA,
            self.MIN_ERROR_CRITERIA,
        )

        flags = cv.CALIB_USE_INTRINSIC_GUESS
        flags |= cv.CALIB_USE_EXTRINSIC_GUESS

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

        return (
            ret,
            camera_matrix_cam1_stereo,
            dist_coefs_cam1_stereo,
            camera_matrix_cam2_stereo,
            dist_coefs_cam2_stereo,
            R,
            T,
        )