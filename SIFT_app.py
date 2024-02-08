#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys


class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height,
                             bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()

        # SIFT - Scale Invariant Feature transform
        # RANSAC - Random Sample Consensus - Separate data points into inliers and outliers

        # read the image into a grey scale cv2 image.
        img = cv2.imread(self.template_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # construct a SIFT object
        sift = cv2.SIFT_create()

        # detect the keypoint in the image,
        # with mask being None, so every part of the image is being searched
        keypoint = sift.detect(gray, None)
        print("the number of key points: ", len(keypoint))

        # draw the keypoint onto the image, show and save it
        # img = cv2.drawKeypoints(gray, keypoint, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("name", img)
        # cv2.imwrite('sift_keypoints.jpg', img)

        # calculate the descriptor for each key point
        descriptor, keypoint = sift.compute(gray, keypoint)


        pixmap = self.convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap)



    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
