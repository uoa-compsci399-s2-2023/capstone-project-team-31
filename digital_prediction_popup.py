from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def __init__(self, parent, original_image, impersonation_image, original_result, impersonation_result):
        self.parent = parent
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.original_label = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.original_label.sizePolicy().hasHeightForWidth())
        self.original_label.setSizePolicy(sizePolicy)
        self.original_label.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.original_label.setFont(font)
        self.original_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_label.setObjectName("original_label")
        self.verticalLayout.addWidget(self.original_label)
        self.original_img = QtWidgets.QLabel()
        self.original_img.setObjectName("original_img")
        self.verticalLayout.addWidget(self.original_img)
        self.original_result = QtWidgets.QLabel()
        self.original_result.setMaximumSize(QtCore.QSize(16777215, 50))
        self.original_result.setObjectName("original_result")
        self.verticalLayout.addWidget(self.original_result)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.imperonsation_label = QtWidgets.QLabel()
        self.imperonsation_label.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.imperonsation_label.setFont(font)
        self.imperonsation_label.setAlignment(QtCore.Qt.AlignCenter)
        self.imperonsation_label.setObjectName("imperonsation_label")
        self.verticalLayout_2.addWidget(self.imperonsation_label)
        self.impersonation_img = QtWidgets.QLabel()
        self.impersonation_img.setObjectName("impersonation_img")
        self.verticalLayout_2.addWidget(self.impersonation_img)
        self.impersonation_result = QtWidgets.QLabel()
        self.impersonation_result.setMaximumSize(QtCore.QSize(16777215, 50))
        self.impersonation_result.setObjectName("impersonation_result")
        self.verticalLayout_2.addWidget(self.impersonation_result)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        QtCore.QMetaObject.connectSlotsByName()

    def set_image(self, original_image, impersonation_image):
        size = self.original_img.size()
        try:
            h, w, ch = image.shape
            bytesPerLine = ch * w
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            convert_to_qt = QtGui.QImage(image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            p = convert_to_qt.scaled(size, QtCore.Qt.KeepAspectRatio)
            
            self.video_display.setPixmap(QtGui.QPixmap.fromImage(p))
        except Exception as e:
            print(e)