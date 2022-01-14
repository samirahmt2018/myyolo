from itertools import count
from warnings import filters
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Permute, LeakyReLU, Conv2D,BatchNormalization,Layer,LeakyReLU,UpSampling2D, Concatenate, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow import keras as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 32, 64, 1],
    (128, 3, 2),
    ["B", 64,128,2],
    (256, 3, 2),
    ["B", 128, 256, 2],#WAS 8
    (512, 3, 2),
    ["B", 256, 512, 2],#WAS 8
    (1024, 3, 2),
    ["B", 512, 1024, 1], #WAS 4 # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]
 

class CNNBlock(Layer):
     def __init__(self, filters, bn_act=True, **kwargs):
         super(CNNBlock,self).__init__()
         self.conv=Conv2D(filters, use_bias=not bn_act,**kwargs)
         self.bn=BatchNormalization()
         self.leaky=LeakyReLU(0.1)
         self.use_bn_act=bn_act
     def call(self,x,training=False): 
         if self.use_bn_act:
             return self.leaky(self.bn(self.conv(x)))
         else:
             return self.conv(x)
class ResidualBlock(Layer):
    def __init__(self, first_filter, second_filter,use_residual=True,num_repeats=1):
        super(ResidualBlock,self).__init__()
        self.res_layers = [];
        for _ in range(num_repeats):
            self.res_layers+=[Sequential(
                [
                CNNBlock(first_filter, kernel_size=1),
                CNNBlock(second_filter, kernel_size=1)
                ]
            )]
        self.use_residual=use_residual
        self.num_repeats=num_repeats
    def call(self,x):
        #print(self.res_layers)
        for layer in self.res_layers:
            x=layer(x)+x if self.use_residual else layer(x)
        return x 
class ScalePrediction(Layer):
    def __init__(self, filters, num_classes):
        super(ScalePrediction,self).__init__()
        self.pred=Sequential(
            [
                CNNBlock(filters,kernel_size=3, padding="same"),
                CNNBlock(3*(num_classes+5), bn_act=False, kernel_size=1),# 5 means [po, x,y,w,h]
            ]
        )
        self.num_classes=num_classes
    def call(self, x):
        #print(x.shape,self.pred(x).shape)
        pp=tf.reshape(self.pred(x), shape=(x.shape[0],3,self.num_classes+5,x.shape[1],x.shape[2] ))
        pp=tf.transpose(pp, perm=[0,1,3,4,2])
        #print("reshape", pp.shape)

        return pp
        #print(pp.shape)
        #return Permute(dims=(1,2,4,5,3))(
        #    tf.reshape(pp, shape=(x.shape[0],3,self.num_classes+5,x.shape[2],x.shape[3] ))
        #)
        #N x 3 x 13 x 13 x 5+num_classes, n no of examples

class YOLOv3(Model):
    def __init__(self, in_channels=3, num_classes=20):
        super(YOLOv3,self).__init__()
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.yolo_layers=self._create_conv_layers()
    def _create_conv_layers(self):
        yolo_layers=[]
        out_channels=0 #this is for inputs for sclae prediction

        for module in config:
            if isinstance(module,tuple):
                filters,kernel_size,stride=module
                yolo_layers.append(CNNBlock(filters, kernel_size=kernel_size, strides=(stride,stride), padding="same" if kernel_size==3 else "valid"))
                out_channels=filters
            elif isinstance(module, list):
                #num_repeats=module[3]
                yolo_layers.append(ResidualBlock(module[1],module[2],num_repeats=module[3]))
                out_channels=module[2]
            elif isinstance(module,str):
                if module == "S":
                    yolo_layers+=[
                        ResidualBlock(out_channels, out_channels//2,use_residual=False, num_repeats=1),
                        CNNBlock(out_channels,kernel_size=1),
                        ScalePrediction(out_channels//2, num_classes=self.num_classes)
                    ]
                elif module == "U":
                    yolo_layers.append(UpSampling2D())
        #print(len(yolo_layers))
        return yolo_layers
    def call(self, x):
        outputs=[]
        route_connections=[]
        count=0
        #print(len(self.yolo_layers))
        for layer in self.yolo_layers:
            count+=1
            
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                #print("scale prediction found at layer ",count)
                continue
            #print(layer)
            x=layer(x)
            #print(x.shape)
            if isinstance(layer, ResidualBlock) and layer.num_repeats==2:#was 8
                route_connections.append(x)
                #print("route connection found at layer ",count)
            elif isinstance(layer,UpSampling2D):
                #print("upsample was detected at layer",count )
                x=tf.concat([x,route_connections[-1]], axis=3)
                route_connections.pop()
        return outputs
    def model(self):
        x=Input(shape=(416,416,3))
        return Model(inputs=[x], outputs=self.call(x))

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    #
    # 
    # model.model().summary()
    #model.build(input_shape=(416,416,3))
    #model.summary()
    x = tf.random.uniform((2,IMAGE_SIZE, IMAGE_SIZE,3))
    out = model(x)
    #print(len(out))
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
    #model=Sequential(
    #    [
    #        CNNBlock(32,kernel_size=3),
    #        ResidualBlock(64,128,use_residual=False),
    #        layers.Flatten(),
    #        layers.Dense(10),
    #    ]
#
    #)
    #model.compile(optimizer=Adam(),loss=categorical_crossentropy)
    #model.build(input_shape=(None,416,416,3))
    model.summary()