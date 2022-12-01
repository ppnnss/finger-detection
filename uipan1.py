
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication
import sys
import cv2 as cv
import numpy as np
from PIL import Image as im
# import matplotlib.pyplot as plt


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
        self.pushButton_4.setGeometry(QtCore.QRect(200, 750, 330, 35))
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

    def clicker1(self):
        self.image1 = QFileDialog.getOpenFileName(MainWindow, "Open File", r"C:\Users\User\Documents\___001Praew's\1-year4\image processing", "PNG Files (*.png);; Jpg Files (*.jpg)")
        self.pixmap1 = QPixmap(self.image1[0])
        # QPixmap.setScaledContents( true );
        self.label_5.setPixmap(self.pixmap1.scaled(self.label_5.width(), self.label_5.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        # print(type(self.image1))
        return self.image1


    def clicker2(self):
        self.image2 = QFileDialog.getOpenFileName(MainWindow, "Open File", r"C:\Users\User\Documents\___001Praew's\1-year4\image processing", "PNG Files (*.png);; Jpg Files (*.jpg)")
        self.pixmap2 = QPixmap(self.image2[0])
        self.label_6.setPixmap(self.pixmap2)
        return self.image2

    def converttothin_cvqt(self, image): 
        image = self.image1 
        self.image = cv.imread(image[0])
        # self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        # height, width = self.image.shape[:2]
        mask = np.zeros(self.image.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        height, width = self.image.shape[:2]
        cv.setRNGSeed(0)
        rect = (15,25,width-20,height-30)  #(x,y,w,h)
        new_mask, fgdModel, bgdModel  = cv.grabCut(self.image, mask, rect, bgdModel, fgdModel,10,cv.GC_INIT_WITH_RECT)
        mask2 = np.where((new_mask==cv.GC_PR_BGD)|(new_mask==cv.GC_BGD),0,1).astype("uint8")
        self.img1 = (self.image)*mask2[:,:,np.newaxis]
        # print((self.img1.shape))
        # print((mask2.shape))
        height1, width1, ch1 = self.img1.shape
        convo = ch1 * width1
        self.img1 = QtGui.QImage(self.img1, width1, height1, convo, QImage.Format_BGR888)
        self.img_resize = QtGui.QPixmap.fromImage(self.img1)
        self.label_7.setPixmap(self.img_resize.scaled(self.label_7.width(), self.label_7.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation) )
       
        cv.waitKey()

    # def clicker4(self):
    #     self.label_9.setText('Test2')
    #     self.label_10.setText('Test2')

    def retranslateUi(self, MainWindow): 
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Fingerprint Detection by Image Processing"))
        self.label.setText(_translate("MainWindow", "Please insert 2 images for the detection"))
        # self.label_2.setText(_translate("MainWindow", "Please insert 2 images for the detection"))
        # self.label_3.setText(_translate("MainWindow", "Result"))
        # self.label_4.setText(_translate("MainWindow", "Match"))
        self.label_5.setText(_translate("MainWindow", "Image 1"))
        self.label_6.setText(_translate("MainWindow", "Image 2"))
        self.label_7.setText(_translate("MainWindow", "Thinned image 1"))
        self.label_8.setText(_translate("MainWindow", "Thinned image 2"))
        self.label_9.setText(_translate("MainWindow", "Output image"))
        self.label_10.setText(_translate("MainWindow", "Match / Not Match"))
        self.pushButton.setText(_translate("MainWindow", "Insert image 1"))
        self.pushButton_2.setText(_translate("MainWindow", "Insert image 2"))
        self.pushButton_3.setText(_translate("MainWindow", "Click to skeletonize the images"))
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
