# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Github\capstone-project-team-31\physical_popup.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setFixedSize(624, 124)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.impersonation_label = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.impersonation_label.sizePolicy().hasHeightForWidth())
        self.impersonation_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.impersonation_label.setFont(font)
        self.impersonation_label.setObjectName("impersonation_label")
        self.gridLayout_2.addWidget(self.impersonation_label, 0, 0, 1, 1)
        self.selection_box = QtWidgets.QComboBox(Form)
        self.selection_box.setMinimumSize(QtCore.QSize(300, 50))
        self.selection_box.setMaximumSize(QtCore.QSize(300, 16777215))
        self.selection_box.setObjectName("selection_box")
        self.selection_box.addItems(["Gender", "Ethnicity", "Emotion"])
        self.gridLayout_2.addWidget(self.selection_box, 0, 1, 1, 1)
        self.select_btn = QtWidgets.QPushButton(Form)
        self.select_btn.setMinimumSize(QtCore.QSize(300, 50))
        self.select_btn.setMaximumSize(QtCore.QSize(300, 16777215))

        self.select_btn.setFont(font)
        self.select_btn.setObjectName("select_btn")
        self.gridLayout_2.addWidget(self.select_btn, 1, 0, 1, 1)
        self.cancel_btn = QtWidgets.QPushButton(Form)
        self.cancel_btn.setMinimumSize(QtCore.QSize(300, 50))
        self.cancel_btn.setMaximumSize(QtCore.QSize(300, 16777215))

        self.cancel_btn.setFont(font)
        self.cancel_btn.setObjectName("cancel_btn")
        self.gridLayout_2.addWidget(self.cancel_btn, 1, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.impersonation_label.setText(_translate("Form", "Impersonation Type"))
        self.select_btn.setText(_translate("Form", "Select"))
        self.cancel_btn.setText(_translate("Form", "Cancel"))
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())