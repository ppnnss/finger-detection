
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication
import sys
import cv2 as cv
import numpy as np


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        # QtWidgets.QMainWindow.__init__(self)
        # self.ui=Ui_MainWindow()
        # self.ui.setupUi(self)
        # uic.loadUi("uipan1.ui", self)

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920,1080) #h,w
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 30, 1000, 55)) #(leftright,updown,boxlength,boxwidth)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.label_5 = QtWidgets.QLabel(self.centralwidget) #input img1
        self.label_5.setGeometry(QtCore.QRect(90, 99, 330, 450))
        self.label_5.setAutoFillBackground(True)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_5.setStyleSheet("border: 3px solid black;")

        self.label_6 = QtWidgets.QLabel(self.centralwidget) #input img2
        self.label_6.setGeometry(QtCore.QRect(460, 99, 330, 450))
        self.label_6.setAutoFillBackground(True)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setStyleSheet("border: 3px solid black;")
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(self.centralwidget) #input img3
        self.label_7.setGeometry(QtCore.QRect(1130, 99, 330, 450))
        self.label_7.setAutoFillBackground(True)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setStyleSheet("border: 3px solid black;")
        self.label_7.setObjectName("label_7")

        self.label_8 = QtWidgets.QLabel(self.centralwidget) #input img4
        self.label_8.setGeometry(QtCore.QRect(1500, 99, 330, 450))
        self.label_8.setAutoFillBackground(True)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setStyleSheet("border: 3px solid black;")
        self.label_8.setObjectName("label_8")

        self.label_9 = QtWidgets.QLabel(self.centralwidget) #output img
        self.label_9.setGeometry(QtCore.QRect(690, 630, 600, 300))
        self.label_9.setAutoFillBackground(True)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setStyleSheet("border: 3px solid black;")
        self.label_9.setObjectName("label_8")

        self.label_10 = QtWidgets.QLabel(self.centralwidget) #match or not match
        self.label_10.setGeometry(QtCore.QRect(1400, 700, 400, 170))
        self.label_10.setAutoFillBackground(True)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setStyleSheet("border: 5px solid green;")
        self.label_10.setObjectName("label_8")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget) #insert img1
        self.pushButton.setGeometry(QtCore.QRect(155, 570, 200, 40))
        self.pushButton.setObjectName("pushButton")
        # self.pushButton.setStyleSheet("background-color: green")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget) #insert img2
        self.pushButton_2.setGeometry(QtCore.QRect(525, 570, 200, 40))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget) #click for skeletonize
        self.pushButton_3.setGeometry(QtCore.QRect(810, 324, 300, 45))
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget) #click for the detection
        self.pushButton_4.setGeometry(QtCore.QRect(200, 750, 300, 40))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setIcon(QIcon('icon.png'))


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.clicker1)
        self.pushButton_2.clicked.connect(self.clicker2)
        self.pushButton_3.clicked.connect(self.converttothin_cvqt)
        self.pushButton_4.clicked.connect(self.matching)

    def clicker1(self):
        self.image1 = QFileDialog.getOpenFileName(MainWindow, "Open File", r"C:\Users\User\Documents\___001Praew's\1-year4\image processing\input" , "PNG Files (*.png);; Jpg Files (*.jpg)")
        self.pixmap1 = QPixmap(self.image1[0])
        self.label_5.setPixmap(self.pixmap1.scaled(self.label_5.width(), self.label_5.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        return self.image1


    def clicker2(self):
        self.image2 = QFileDialog.getOpenFileName(MainWindow, "Open File", r"C:\Users\User\Documents\___001Praew's\1-year4\image processing\input", "PNG Files (*.png);; Jpg Files (*.jpg)")
        self.pixmap2 = QPixmap(self.image2[0])
        self.label_6.setPixmap(self.pixmap2.scaled(self.label_6.width(), self.label_6.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        return self.image2

    def converttothin_cvqt(self): 
        #input image 1 operation
        self.image = cv.imread(self.image1[0])
        mask = np.zeros(self.image.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        height, width, _ = self.image.shape   
        cv.setRNGSeed(0)
        rect = (15,25,width-20,height-30)  #(x,y,w,h)
        new_mask, fgdModel, bgdModel  = cv.grabCut(self.image, mask, rect, bgdModel, fgdModel,10,cv.GC_INIT_WITH_RECT)
        mask2 = np.where((new_mask==cv.GC_PR_BGD)|(new_mask==cv.GC_BGD),0,1).astype("uint8")
        self.grabcut = (self.image)*mask2[:,:,np.newaxis]
        background = cv.absdiff(self.image, self.grabcut)
        background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]
        outremove = background + self.grabcut
        gray1 = cv.cvtColor(outremove, cv.COLOR_BGR2GRAY)
        img_blur = cv.GaussianBlur(gray1, (5,5), 0)
        img_canny = cv.Canny(img_blur, 20, 10)
        sharp33 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp55 = np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,9,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]])
        img_sharp = cv.filter2D(img_canny, -1, sharp33)
        ret, thresh = cv.threshold(img_sharp, 127, 255, cv.THRESH_BINARY)
        thin = cv.ximgproc.thinning(thresh)
        sift = cv.xfeatures2d.SIFT_create()
        kp, self.des = sift.detectAndCompute(thin, None)
        keypoint = cv.drawKeypoints(thin, kp, thin, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
        h, w, ch = keypoint.shape
        convo = ch * w
        self.kp = QtGui.QImage(keypoint, w, h, convo, QImage.Format_BGR888)
        self.kp1 = QtGui.QPixmap.fromImage(self.kp)
        self.label_7.setPixmap(self.kp1.scaled(self.label_7.width(), self.label_7.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation) )

        #input image 2 operation
        self.image2 = cv.imread(self.image2[0])
        mask2 = np.zeros(self.image2.shape[:2],np.uint8)
        bgdModel2 = np.zeros((1,65),np.float64)
        fgdModel2 = np.zeros((1,65),np.float64)
        height2, width2, ch2 = self.image2.shape   
        cv.setRNGSeed(0)
        rect2 = (15,25,width2-20,height2-30)  #(x,y,w,h)
        new_mask2, fgdModel2, bgdModel2  = cv.grabCut(self.image2, mask2, rect2, bgdModel2, fgdModel2,10,cv.GC_INIT_WITH_RECT)
        masknew = np.where((new_mask2==cv.GC_PR_BGD)|(new_mask2==cv.GC_BGD),0,1).astype("uint8")
        self.grabcut2 = (self.image2)*masknew[:,:,np.newaxis]
        background2 = cv.absdiff(self.image2, self.grabcut2)
        background2[np.where((background2 > [0,0,0]).all(axis = 2))] = [255,255,255]
        outremove2 = background2 + self.grabcut2
        gray2 = cv.cvtColor(outremove2, cv.COLOR_BGR2GRAY)
        img_blur2 = cv.GaussianBlur(gray2, (5,5), 0)
        img_canny2 = cv.Canny(img_blur2, 20, 10)
        sharp33_2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp55_2 = np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,9,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]])
        img_sharp2 = cv.filter2D(img_canny2, -1, sharp33_2)
        ret2, thresh2 = cv.threshold(img_sharp2, 127, 255, cv.THRESH_BINARY)
        thin2 = cv.ximgproc.thinning(thresh2)
        sift2 = cv.xfeatures2d.SIFT_create()
        kp2, self.des2 = sift2.detectAndCompute(thin2, None)
        keypoint2 = cv.drawKeypoints(thin2, kp2, thin2, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
        h2, w2, ch2 = keypoint2.shape
        convo2 = ch2 * w2
        self.kp2 = QtGui.QImage(keypoint2, w2, h2, convo2, QImage.Format_BGR888)
        self.kp_2 = QtGui.QPixmap.fromImage(self.kp2)
        self.label_8.setPixmap(self.kp_2.scaled(self.label_8.width(), self.label_8.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation) )

        return self.des, self.des2

    def matching(self):
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)  
        nn_match = matcher.knnMatch(self.des, self.des2, 2)
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        matches = sorted(bf.match(self.des, self.des2), key= lambda match:match.distance)

    def retranslateUi(self, MainWindow): 
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Fingerprint Detection by Image Processing"))
        self.label.setText(_translate("MainWindow", "Please insert 2 images for the detection"))
        self.label_5.setText(_translate("MainWindow", "Image 1"))
        self.label_6.setText(_translate("MainWindow", "Image 2"))
        self.label_7.setText(_translate("MainWindow", "Thinned image 1"))
        self.label_8.setText(_translate("MainWindow", "Thinned image 2"))
        self.label_9.setText(_translate("MainWindow", "Output image"))
        self.label_10.setText(_translate("MainWindow", "Match / Not Match"))
        self.pushButton.setText(_translate("MainWindow", "Insert image 1"))
        self.pushButton_2.setText(_translate("MainWindow", "Insert image 2"))
        self.pushButton_3.setText(_translate("MainWindow", "Convert to thin and find the keypoint"))
        self.pushButton_4.setText(_translate("MainWindow", "Start the detection process"))

        self.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    # ui.setupUi(MainWindow)
    MainWindow.show()
    app.exec_()

# app = QApplication(sys.argv)
# UIwindow = Ui_MainWindow()
# app.exec_()