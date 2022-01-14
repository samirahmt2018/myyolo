"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

from numpy import False_
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers,Model
from tensorflow.python.keras.engine.input_layer import Input

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size,name="cnnblock", strides=2,padding="same"):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size,strides=strides, padding=padding, name=name)
        self.batchnorm = layers.BatchNormalization()
        self.leakyrelu = layers.LeakyReLU(0.1)

    def call(self, input_tensor,training=False):
        x=self.conv(input_tensor)
        x=self.batchnorm(x,training=training)
        x=self.leakyrelu(x)
        return x

class Yolov1(Model):
    def __init__(self, num_classes=10):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.num_classes = num_classes
        self.block1=CNNBlock(64, 7, strides=2,padding="same", name="block01")
        self.block2=CNNBlock(192, 3, strides=1, padding="same", name="block02")
        self.block3=CNNBlock(128, 1, strides=1, padding="same", name="block03")
        self.block4=CNNBlock(256, 3, strides=1, padding="same", name="block04")
        self.block5=CNNBlock(256, 1, strides=1, padding="same", name="block05")
        self.block6=CNNBlock(512, 3, strides=1, padding="same", name="block06")
        ##repetitions(4x
        self.block7=CNNBlock(256, 1, strides=1, padding="same", name="block07")
        self.block8=CNNBlock(512, 3, strides=1, padding="same", name="block08")
        self.block9=CNNBlock(256, 1, strides=1, padding="same", name="block09")
        self.block10=CNNBlock(512, 3, strides=1, padding="same", name="block10")
        self.block11=CNNBlock(256, 1, strides=1, padding="same", name="block11")
        self.block12=CNNBlock(512, 3, strides=1, padding="same", name="block12")
        self.block13=CNNBlock(256, 1, strides=1, padding="same", name="block13")
        self.block14=CNNBlock(512, 3, strides=1, padding="same", name="block14")
        ##singles
        self.block15=CNNBlock(512, 1, strides=1, padding="same", name="block15")
        self.block16=CNNBlock(1024, 3, strides=1, padding="same", name="block16")

        #repetitions(2x)
        self.block17=CNNBlock(512, 1, strides=1, padding="same", name="block17")
        self.block18=CNNBlock(1024, 3, strides=1, padding="same", name="block18")
        self.block19=CNNBlock(512, 1, strides=1, padding="same", name="block19")
        self.block20=CNNBlock(1024, 3, strides=1, padding="same", name="block20")

        #final layers
        self.block21=CNNBlock(1024, 3, strides=1, padding="same", name="block21")
        self.block22=CNNBlock(1024, 3, strides=2, padding="same", name="block22")
        self.block23=CNNBlock(1024, 3, strides=1, padding="same", name="block23")
        self.block24=CNNBlock(1024, 3, strides=1, padding="same", name="block24")
        
        #self.fcs = self._create_fcs(**kwargs)

    def call(self, x, S=7,B=2,training=False):
        C=self.num_classes
        x=self.block1(x, training=training)
        x=layers.MaxPool2D()(x)
        x=self.block2(x, training=training)
        x=layers.MaxPool2D()(x)
        x=self.block3(x, training=training)
        x=self.block4(x, training=training)
        x=self.block5(x, training=training)
        x=self.block6(x, training=training)
        x=layers.MaxPool2D()(x)
        x=self.block7(x, training=training)
        x=self.block8(x, training=training)
        x=self.block9(x, training=training)
        x=self.block10(x, training=training)
        x=self.block11(x, training=training)
        x=self.block12(x, training=training)
        x=self.block13(x, training=training)
        x=self.block14(x, training=training)
        x=self.block15(x, training=training)
        x=self.block16(x, training=training)
        x=layers.MaxPool2D()(x)
        x=self.block17(x, training=training)
        x=self.block18(x, training=training)
        x=self.block19(x, training=training)
        x=self.block20(x, training=training)
        x=self.block21(x, training=training)
        x=self.block22(x, training=training)
        x=self.block23(x, training=training)
        x=self.block24(x, training=training)
        x=layers.Flatten()(x)
        x=layers.Dense(496)(x)
        x=layers.Dropout(0.0)(x)
        x=layers.LeakyReLU(0.1)(x)
        x=layers.Dense(S*S*(C+B*5))(x)

        
        return x
    def model(self):
        x=Input(shape=(448,448,3))
        return Model(inputs=[x], outputs=self.call(x))

#def test(S=7, B=2, C=20):
#    model=Yolov1(split_size=S, num_boxes=B, num_classes=C)
#    x=torch.randn((2,3,448,448))
#    print(model(x).shape)
#test()
model=Yolov1(num_classes=3)
model.model().summary()

