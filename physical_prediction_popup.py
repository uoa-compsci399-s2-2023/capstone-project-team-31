# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Github\capstone-project-team-31\physical_prediction_popup.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_(QtWidgets.QWidget):
    def __init__(self, image, result):
        super().__init__()
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.title_label = QtWidgets.QLabel()
        self.title_label.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.title_label.setFont(font)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setObjectName("title_label")
        self.title_label.setText("Physical Attack Prediction")
        self.verticalLayout_2.addWidget(self.title_label)
        self.image_disp = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_disp.sizePolicy().hasHeightForWidth())
        self.image_disp.setSizePolicy(sizePolicy)
        self.image_disp.setMinimumSize(QtCore.QSize(720, 720))
        self.image_disp.setAlignment(QtCore.Qt.AlignCenter)
        self.image_disp.setObjectName("image_disp")
        self.verticalLayout_2.addWidget(self.image_disp)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.prediction_label = QtWidgets.QLabel()
        self.prediction_label.setMinimumSize(QtCore.QSize(0, 50))
        self.prediction_label.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.prediction_label.setFont(font)
        self.prediction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.prediction_label.setObjectName("prediction_label")
        self.prediction_label.setText("Prediction Result:")
        self.horizontalLayout.addWidget(self.prediction_label)
        self.result_disp = QtWidgets.QLabel()
        self.result_disp.setMinimumSize(QtCore.QSize(0, 50))
        self.result_disp.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.result_disp.setFont(font)
        self.result_disp.setText(result)
        self.result_disp.setAlignment(QtCore.Qt.AlignCenter)
        self.result_disp.setObjectName("result_disp")
        self.horizontalLayout.addWidget(self.result_disp)
        self.verticalLayout_2.addLayout(self.horizontalLayout)


    def set_image(self, image):
        self.image_disp.setPixmap(image)
        

