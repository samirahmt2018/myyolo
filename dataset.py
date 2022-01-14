"""
Creates a Pytf dataset to load the Pascal VOC & MS COCO datasets
"""

from __future__ import annotations
import sys
import config
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend  as K
from PIL import Image, ImageFile
from tensorflow.keras.utils import Sequence
from myutils import preprocess_true_boxes,plot_image,get_anchors,iou_width_height as iou,cells_to_bboxes, non_max_suppression as nms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Sequence):
    def __init__(
        self,
        annotation_file,
        class_file,
        img_dir,
        anchors_path="yolo_anchors.txt",
        image_size=416,
        S=[13, 26, 52],#grid sizes [13,26,53]
        C=20,
        transform=None,
        n_channels=3,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        max_boxes=20,


    ):
        self.dim = [image_size,image_size,n_channels]
        self.batch_size = batch_size
        self.max_boxes=20
        self.n_channels = n_channels
        self.input_shape=[image_size,image_size]
        self.shuffle = shuffle
        self.annotations = open(annotation_file).readlines()
        self.list_IDs = range(len(self.annotations))
        self.indexes = range(len(self.annotations))
        self.img_dir = img_dir
        self.classes = open(class_file).readlines()
        self.n_classes = len(self.classes)
        self.image_size = image_size
        self.transform = transform
        self.S = S 
        self.anchors =  get_anchors(anchors_path)  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return int(np.floor(len(self.annotations)/self.batch_size))

    def __get_single_item(self, index):
        
        line = self.annotations[index].split(" ")
        #print(line)
        image = Image.open(self.img_dir+ line[0])
        iw, ih = image.size
        h, w = (self.image_size,self.image_size)
        bbboxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        anchors = config.ANCHORS
        anchors=K.constant(anchors[0] + anchors[1] + anchors[2])
        
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        ##dx = (w-nw)//2
        ##dy = (h-nh)//2
        image = np.array(image.resize((nw,nh), Image.BICUBIC))/255
        targets=[np.zeros((self.num_anchors//3,S,S,6)) for S in self.S] #[p_of_targets, x,y,w,h,class]
        #print(bbboxes.shape)
        bbboxes=np.float32(bbboxes)
        nbox=np.zeros((5,))
        bcount=0
        per_bcount=0
        all_bcount=0
        for box in  bbboxes:
            bcount+=1
            #print(f"box {bcount} found")
            ##new box style for Aladdin Persson, initially was in xmin,xmax,ymin,ymax,class
            #print(box,box.shape)
            nbox[2]=(box[2]-box[0])/iw
            nbox[3]=(box[3]-box[1])/ih
            nbox[0]=(box[2]+box[0])*0.5/iw
            nbox[1]=(box[3]+box[1])*0.5/ih
            nbox[4]=box[4]
            # print(box,"boxshape",box[2:4])
            iou_anchors=iou(K.constant(nbox[2:4]),anchors)
            anchor_indices=tf.argsort(iou_anchors,direction="DESCENDING")
            #==print(anchor_indices)
            x,y,width,height,class_label=nbox
            has_anchor=[False,False,False]
            #print("class",self.classes[int(class_label)],class_label)
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale #which scale
                anchor_on_scale= anchor_idx%self.num_anchors_per_scale # which anchor on the particular anchor
                S = self.S[scale_idx]
                i,j = int(S*y), int(S*x)  #which cell index x,y located
                anchor_taken= targets[scale_idx][anchor_on_scale,i,j,0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale,i,j,0]=1
                    x_cell,y_cell=S*x-j , S*y-i
                    width_cell, height_cell=(width*S, height*S)
                    box_coordinates = tf.constant([x_cell,y_cell,width_cell,height_cell])
                    targets[scale_idx][anchor_on_scale,i,j,1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale,i,j,5] = int(class_label)
                    per_bcount+=1
                    has_anchor[scale_idx]=True
                    #print(targets[scale_idx][anchor_on_scale,i,j])
                    #print(f"Correct Anchor was found for box {bcount}:{per_bcount} anchor {anchor_idx},{int(i)},{int(j)},{S}")
                elif not anchor_taken and iou_anchors[anchor_idx] >self.ignore_iou_thresh:
                    all_bcount+=1
                    targets[scale_idx][anchor_on_scale,i,j,0] = -1 #ignoring this prediction
                    #print(f"Incorrect Anchor was found for box {bcount}:{all_bcount} anchor {anchor_idx}")
             
        #print(targets[0])
        #targets=tf.ragged.constant(targets)
        return image, targets
              
 
        # correct boxes
        #box_data = np.zeros((box.shape))
        #box_data = np.zeros((self.max_boxes,5))
        #if len(box)>0:
        #    np.random.shuffle(box)
        #    if len(box)>self.max_boxes: box = box[:self.max_boxes]
        #    box[:, [0,2]] = box[:, [0,2]]*scale ##+ dx
        #    box[:, [1,3]] = box[:, [1,3]]*scale ##+ dy
        #    box_data[:len(box)] = box

        #return image, box_data

    def __get_data(self,batch=1, batch_size=1):
        for i in range(batch*batch_size, batch*batch_size+batch_size):
            print("ok")
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X=[]
        #box_data = []
       
        y=[np.empty((self.batch_size,self.num_anchors//3,S,S,6)) for S in self.S] #[p_of_targets, x,y,w,h,class]
        
                
                
        X= np.empty((self.batch_size, *self.dim))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #print(list_IDs_temp)
            #print("ind",self.indexes[ID],i,y[0].shape)
            
            X[i,],y_temp = self.__get_single_item(self.indexes[ID])
            #print(y_temp[0].shape,y[0].shape)
            for j in range(len(self.S)):
                y[j][i]=y_temp[j]
                    
        return X, y
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    
    def augment(x,y):
        image=tf.image.random_brightness(x,max_delta=0.05)
        return image,y

    def train_generator(self):
        while True:
            for start in range(0, len(self.indexes), self.batch_size):
                #x_batch = []
                y_batch=[np.empty((self.batch_size,self.num_anchors//3,S,S,6)) for S in self.S] #[p_of_targets, x,y,w,h,class]
        
                
                
                x_batch= np.empty((self.batch_size, *self.dim))

                end = min(start + self.batch_size-1, len(self.annotations))
                #print(0,len(self.S),len(y_batch[0]),y_batch[0].shape)
                #print(y_batch[0:len(self.S)][0].shape)
                for i,step in enumerate(range(start, end)):
                    #print("indexes", len(self.indexes),i, self.indexes[step],start,end )
                    x_batch[i,],y_batch[0:len(self.S)][i] = self.__get_single_item(self.indexes[step])
                    #print("Y generated",y_temp[0].shape)
                    #y_temp=tf.expand_dims(y_temp,0)
                    #y_batch.append(y_temp)
                
                yield (x_batch, y_batch)

def test():
    
    transform = config.test_transforms

    dataset = YOLODataset(
        "/Volumes/MLData/Python/Breast_Detector/annotation_new.txt",
        "/Volumes/MLData/Python/Breast_Detector/classes.txt",
        "/Volumes/MLData/Python/Breast_Detector/yolo_processed/",
        S=[13, 26, 52],
        anchors_path="yolo_anchors.txt",
        transform=transform,
        batch_size=1
    )
    scaling_S=tf.constant([[[13, 13],
         [13, 13],
         [13, 13]],

        [[26, 26],
         [26, 26],
         [26, 26]],

        [[52, 52],
         [52, 52],
         [52, 52]]], dtype=tf.float32)
   
    #print(tf.constant(config.ANCHORS).shape)
    #print(tf.repeat(tf.expand_dims(tf.expand_dims(tf.constant(config.S,dtype=float32),1),2),(1, 3, 2)))
    scaled_anchors = (
        tf.constant(config.ANCHORS)
        * scaling_S
    )
    #print(scaled_anchors)
    for x, y in dataset:
        boxes = np.empty([1,6])
        #print(y[0].shape)
        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            #print(anchor.shape)
            #print(y[i].shape)
            c_box=cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor,
            )
            #print(c_box[0].shape)
            boxes=np.vstack([boxes,c_box[0]])
        #print(boxes)
        boxes = nms(boxes.tolist(), iou_threshold=1, threshold=0.7, box_format="midpoint")
        #print(boxes)
        plot_image(x[0], boxes,dataset.classes)
    #print(boxes)
    #plot_image(x, y, dataset.classes)
if __name__ == "__main__":
    test()


