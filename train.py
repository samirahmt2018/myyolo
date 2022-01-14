"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

from numpy import float32
from tensorflow.python.keras.engine import training
import config
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from yolo3 import YOLOv3
from tqdm import tqdm
import tensorboard as tb
#from myutils import (
#    mean_average_precision,
#    cells_to_bboxes,
#    get_evaluation_bboxes,
#    save_checkpoint,
#    load_checkpoint,
#    check_class_accuracy,
#    get_loaders,
#    plot_couple_examples
#)
from loss import myYoloLoss
from dataset import YOLODataset
import warnings
warnings.filterwarnings("ignore")


def train_fn(data_train,data_valid, model, optimizer, scaled_anchors,num_epochs):
   
    losses = []
    min_loss=1e10
    for epoch in range(num_epochs):
        print(f"\n Start of Training Epoch {epoch}") 
        for step, (x_batch, y_batch) in tqdm(enumerate(data_train),"Training Steps",total=data_train.__len__()):
            
            with tf.GradientTape() as tape:
                #print("ini batch",x_batch.shape)
                y_pred= model(x_batch, training=True)
                #print("third dim",y_pred[2].shape, y_batch[2].shape)
                #print("second dim loss", myYoloLoss(y_pred[1], y_batch[1], scaled_anchors[1]) )
                #loss= myYoloLoss(y_pred,y_batch,)
                #print("third dim loss", myYoloLoss(y_pred[2], y_batch[2], scaled_anchors[2]))
                loss = (
                            myYoloLoss(y_pred[0], y_batch[0], scaled_anchors[0])
                            + myYoloLoss(y_pred[1], y_batch[1], scaled_anchors[1])
                            + myYoloLoss(y_pred[2], y_batch[2], scaled_anchors[2])
                        )
                #print(loss)
            losses.append(loss)

            gradients= tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients,model.trainable_weights))
        val_losses=[]
        for step, (x_batch, y_batch) in tqdm(enumerate(data_valid),desc="Validation Steps",total=data_valid.__len__()):
            
            
            y_pred= model(x_batch, training=True)
            #print("third dim",y_pred[2].shape, y_batch[2].shape)
            #print("second dim loss", myYoloLoss(y_pred[1], y_batch[1], scaled_anchors[1]) )
            #loss= myYoloLoss(y_pred,y_batch,)
            #print("third dim loss", myYoloLoss(y_pred[2], y_batch[2], scaled_anchors[2]))
            val_loss = (
                        myYoloLoss(y_pred[0], y_batch[0], scaled_anchors[0])
                        + myYoloLoss(y_pred[1], y_batch[1], scaled_anchors[1])
                        + myYoloLoss(y_pred[2], y_batch[2], scaled_anchors[2])
                    )
            #print(loss)
            val_losses.append(val_loss)

        
#
        ## update progress bar
        mean_loss = sum(losses) / len(losses)
        mean_val_loss=sum(val_losses)/len(val_losses)
        if mean_val_loss<min_loss:
            min_loss=mean_val_loss
            model.save_weights("models/model_"+str(int(min_loss))+".h5")
        #if epoch==0:
        #    model.summary()
        print(f"Epoch: {epoch} Loss:{float(mean_loss)}  Validation Loss:{float(mean_val_loss)}")
        #print(f"Mean loss over epoch{mean_loss}")
        



def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES)
   
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    
    
    train_generator = YOLODataset(
        config.ANN_PATH,
        config.CLASS_PATH,
        config.IMG_DIR,
        S=[13, 26, 52],
        anchors_path="yolo_anchors.txt",
    )
    valid_generator = YOLODataset(
        config.VAL_PATH,
        config.CLASS_PATH,
        config.IMG_DIR,
        S=[13, 26, 52],
        anchors_path="yolo_anchors.txt",
    )
   
    scaling_S=tf.constant([[[13, 13],
         [13, 13],
         [13, 13]],

        [[26, 26],
         [26, 26],
         [26, 26]],

        [[52, 52],
         [52, 52],
         [52, 52]]], dtype=float32)
   
    #print(tf.constant(config.ANCHORS).shape)
    #print(tf.repeat(tf.expand_dims(tf.expand_dims(tf.constant(config.S,dtype=float32),1),2),(1, 3, 2)))
    scaled_anchors = (
        tf.constant(config.ANCHORS)
        * scaling_S
    )

    
    train_fn(train_generator,valid_generator, model, optimizer, scaled_anchors,config.NUM_EPOCHS)

        #if config.SAVE_MODEL:
        #    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

       

if __name__ == "__main__":
    main()
