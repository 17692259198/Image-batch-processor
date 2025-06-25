#'''
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QBrush,QPixmap

import imageRename
import imageMark
class Ui_MainWindow(QtWidgets.QWidget):

    # 自动生成的代码，用来对窗体进行设置
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600) #设置窗体大小
        # 设置菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 813, 23))
        self.menubar.setObjectName("menubar")
        # 添加“主菜单”菜单
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        # 添加“关于”菜单
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")

        # 添加“添加水印”菜单
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        # 添加“重命名”菜单
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")



        # 添加“添加水印”子菜单
        self.actionMark = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon() # 创建图标对象
        # 设置图标文件
        icon.addPixmap(QtGui.QPixmap("img/mark.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionMark.setIcon(icon) # 为“添加水印”子菜单设置图标
        self.actionMark.setObjectName("actionMark")
        # 添加“批量重命名”子菜单
        self.actionRename = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon() # 创建图标对象
        # 设置图标文件
        icon1.addPixmap(QtGui.QPixmap("img/rename.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRename.setIcon(icon1) # 为“批量重命名”子菜单设置图标
        self.actionRename.setObjectName("actionRename")
        # 将“添加水印”子菜单添加到“主菜单”菜单中
        self.menu.addAction(self.actionMark)
        # 将“批量重命名”子菜单添加到“主菜单”菜单中
        self.menu.addAction(self.actionRename)

        # 将“添加水印”子菜单添加到“主菜单”菜单中
        self.menu_3.addAction(self.actionMark)
        # 将“批量重命名”子菜单添加到“主菜单”菜单中
        self.menu_4.addAction(self.actionRename)


        # 菜单栏中添加“主菜单”
        self.menubar.addAction(self.menu.menuAction())
        # 添加“关于本软件”子菜单
        self.actionAbout = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon() # 创建图标对象
        # 设置图标文件
        icon.addPixmap(QtGui.QPixmap("img/about.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAbout.setIcon(icon) # 为“关于本软件”子菜单设置图标
        self.actionAbout.setObjectName("actionAbout")
        # 将“关于本软件”子菜单添加到“关于”菜单中
        self.menu_2.addAction(self.actionAbout)
        # 菜单栏中添加“关于”菜单
        self.menubar.addAction(self.menu_2.menuAction())

        # 菜单栏中添加“添加水印”
        self.menubar.addAction(self.menu_3.menuAction())
        # 菜单栏中添加“重命名”
        self.menubar.addAction(self.menu_4.menuAction())


        # 设置窗体背景
        palette = QtGui.QPalette()
        # 设置窗体背景自适应
        palette.setBrush(MainWindow.backgroundRole(),QBrush(QPixmap("img/2.png").scaled(MainWindow.size(),QtCore.Qt.IgnoreAspectRatio,QtCore.Qt.SmoothTransformation)))
        MainWindow.setPalette(palette)
        MainWindow.setAutoFillBackground(True) # 设置自动填充背景
        # 禁止显示最大化按钮及调整窗体大小
        MainWindow.setFixedSize(800, 600);
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 创建第一个按钮并设置图标
        self.pushButton = QPushButton(MainWindow)
        self.pushButton.setText("添加水印")
        self.pushButton.setGeometry(10, 90, 100, 50)
        icon3 = QtGui.QIcon()  # 创建图标对象
        icon3.addPixmap(QtGui.QPixmap("img/1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon3)  # 为“添加水印”子菜单设置图标

        # 创建第二个按钮并设置图标
        self.pushButton2 = QPushButton(MainWindow)
        self.pushButton2.setText("重命名")
        self.pushButton2.setGeometry(10, 30, 100, 50)
        icon4 = QtGui.QIcon()  # 创建图标对象
        icon4.addPixmap(QtGui.QPixmap("img/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton2.setIcon(icon4)  # 为“批量重命名”子菜单设置图标



        # 连接按钮的点击信号到槽函数
        self.pushButton.clicked.connect(self.openMark)
        self.pushButton2.clicked.connect(self.openRename)


    # 自动生成的代码，用来设置窗体中控件的默认值
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图片批量处理器"))
        self.menu.setTitle(_translate("MainWindow", "主菜单"))
        self.menu_2.setTitle(_translate("MainWindow", "|| 关于"))

        self.menu_3.setTitle(_translate("MainWindow", "||添加水印"))
        self.menu_4.setTitle(_translate("MainWindow", "|| 重命名"))

        self.actionMark.setText(_translate("MainWindow", "添加水印"))
        self.actionRename.setText(_translate("MainWindow", "批量重命名"))
        self.actionAbout.setText(_translate("MainWindow", "关于本软件"))
        # 关联“添加水印”菜单的方法
        self.actionMark.triggered.connect(self.openMark)
        # 关联“批量重命名”菜单的方法
        self.actionRename.triggered.connect(self.openRename)
        # 关联“关于本软件”菜单的方法
        self.actionAbout.triggered.connect(self.about)


    # 打开水印窗体
    def openMark(self):
        self.another = imageMark.Ui_MarkWindow()  # 创建水印窗体对象
        self.another.show()  # 显示窗体

    # 打开重命名窗体
    def openRename(self):
        self.another = imageRename.Ui_RenameWindow()  # 创建重命名窗体对象
        self.another.show()  # 显示窗体


    # 关于本软件
    def about(self):
        QMessageBox.information(None, '关于本软件', '图片批量处理器是一款提供日常工作的工具软件，'
                                               '通过该软件，您可以方便的为图片添加文字水印和图片水印，'
                                               '而且可以自定义添加位置，以及透明度；另外，您还可以通过'
                                               '该软件对图片文件进行重命名，支持文件名大写、小写，以及'
                                               '根据自定义模板对图片文件进行编号。', QMessageBox.Ok)




# 主方法
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow() # 创建窗体对象
    ui = Ui_MainWindow() # 创建PyQt5设计的窗体对象
    ui.setupUi(MainWindow) # 调用PyQt5窗体的方法对窗体对象进行初始化设置
    MainWindow.show() # 显示窗体

    sys.exit(app.exec_()) # 程序关闭时退出进程

#'''
'''
#案例二  房价预测
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 数据预处理：标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# 自定义Callback来记录每个epoch的验证MAE
class MAEHistory(Callback):
    def on_epoch_end(self, epoch, logs={}):
        val_mae = self.model.evaluate(val_data, val_targets, verbose=0)[1]
        logs['val_mae'] = val_mae

  

val_split = 0.2
num_val_samples = int(val_split * len(train_data))
val_data = train_data[-num_val_samples:]
val_targets = train_targets[-num_val_samples:]
train_data = train_data[:-num_val_samples]
train_targets = train_targets[:-num_val_samples]


# 构建模型
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# 训练模型并记录每个epoch的验证MAE
model = build_model()
mae_history = MAEHistory()
history = model.fit(train_data, train_targets, epochs=100, batch_size=16, verbose=1,
                    validation_data=(val_data, val_targets), callbacks=[mae_history])

# 绘制验证MAE随时间（epoch）变化的图像
plt.plot(history.history['val_mae'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Validation MAE over epochs')
plt.show()

# 在测试集上评估模型
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(f'Test MSE: {test_mse_score}, Test MAE: {test_mae_score}')
'''
'''
#案例二  猫狗分类

from keras import layers
from keras import models
from keras import layers, models
from keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#keras的序贯模型
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))
#全连接
model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop( learning_rate=1e-4),
              metrics=['acc'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 所有图像将按1/255重新缩放
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,  # 随机旋转度数
    width_shift_range=0.2,  # 宽度平移范围
    height_shift_range=0.2,
    shear_range=0.2,  # 剪切强度
    zoom_range=0.2,  # 随机缩放范围
    horizontal_flip=True,  # 水平翻转
    vertical_flip=False,  # 垂直翻转
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)


train_dir = "D:/imageMS-master/imageMS/cat_and_dog_small/train"

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # 所有图像将调整为150x150
    batch_size=20,
    class_mode='binary'
)
validation_dir = "D:/imageMS-master/imageMS/cat_and_dog_small/validation"
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
#查看上面对于图片预处理的处理结果
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
#模型训练过程

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator)
#对于模型进行评估，查看预测的准确性
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 保存模型
#model.save("cats_and_dogs_small_2.h5")

'''
#预测
'''
import numpy as np
import cv2
import tensorflow as tf

# 加载模型
new_model = tf.keras.models.load_model('D:/imageMS-master/imageMS/cats_and_dogs_small_2.h5')
new_model.summary()

# 读取图片并调整大小
image = cv2.imread('D:/imageMS-master/imageMS/cat_and_dog_small/test/dogs/dog.1500.jpg')
test_img = cv2.resize(image, (150, 150))
test_img_flat = test_img.reshape(1, -1)[:, :8192]
# 归一化
test_img_flat = test_img_flat.astype('float32') / 255.0

# 进行预测
predict = new_model.predict(test_img_flat)

# 输出预测结果并进行分类
if abs(predict[0] - 0.5) < 0.05:
    print("预测错误")
    title = "can not predict"
else:
    if predict[0] > 0.5:
        print("这是狗")
        title = "it is a Dog"
    else:
        print("这是猫")
        title = "it is a Cat"


cv2.imshow(title, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
#qt
'''
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import tensorflow as tf


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Cat and Dog Classifier')
        self.setGeometry(100, 100, 800, 600)

        # 布局和控件
        layout = QVBoxLayout()

        # 图像显示标签
        self.imageLabel = QLabel()
        self.imageLabel.setFixedSize(800, 600)

        # 预测按钮
        self.predictButton = QPushButton('预测')
        self.predictButton.clicked.connect(self.predict_image)

        # 将控件添加到布局中
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.predictButton)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.load_and_display_image('D:/imageMS-master/imageMS/img/back.png')

        # 加载模型
        self.model = tf.keras.models.load_model('D:/imageMS-master/imageMS/cats_and_dogs_small_2.h5')

    def load_and_display_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
        height, width, channel = image.shape
        qImg = QImage(image.data, width, height, width * channel, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        self.imageLabel.setPixmap(pixmap)

    def predict_image(self):
        image_path = 'D:/imageMS-master/imageMS/cat_and_dog_small/test/dogs/dog.1500.jpg'
        image = cv2.imread(image_path)
        image = cv2.resize(image, (150, 150))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # 增加批次维度

        # 进行预测
        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction)

        # 输出预测结果
        if predicted_class == 0:
            print("这是猫")
            title = "it is a Cat"
        else:
            print("这是狗")
            title = "it is a Dog"

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
'''

