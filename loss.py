from numpy import float32
import tensorflow as tf
from tensorflow.keras.losses import Loss,BinaryCrossentropy
from tensorflow.python.keras.losses import CategoricalCrossentropy, MeanSquaredError, SparseCategoricalCrossentropy
from myutils import intersection_over_union
import config
def myYoloLoss(predictions,target,anchors):
    #constants
    lambda_class=1
    lambda_noobj=10
    lambda_obj=1
    lambda_box=10
    mse=MeanSquaredError()
    entropy=SparseCategoricalCrossentropy(from_logits=True)
    bce=BinaryCrossentropy(from_logits=True)

    target=tf.cast(target,float32)
    #print(predictions.shape,target.shape)
    obj=target[...,0] ==1
    noobj = target[...,0] == 0 ##here we are ignoring the ignored predictins(-1s)

    #no obj loss
    no_object_loss = bce(predictions[...,0:1][noobj], target[...,0:1][noobj])


    #object Loss
    anchors = tf.reshape(anchors,(1,3,1,1,2)) # TO MATCH THE DIMN OF HEIGHT AND WIDTHTHREE ANCHORS FOR PARTICULAR 
   
    box_preds = tf.concat([tf.sigmoid(predictions[..., 1:3]), tf.exp(predictions[..., 3:5]) * anchors], axis=-1)
    #print("boxpredshaoe", box_preds[obj].shape, target[..., 1:5][obj].shape)
    ious = tf.stop_gradient(intersection_over_union(box_preds[obj], target[..., 1:5][obj]))
    
    object_loss = mse(tf.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

    # ======================== #
    #   FOR BOX COORDINATES    #
    # ======================== #

    predictions=tf.Variable(predictions)
    target=tf.Variable(target)
    predictions_temp = tf.sigmoid(predictions[..., 1:3])  # x,y coord inates to be in 0 and 
    target_temp = tf.math.log(
        (1e-16 + target[..., 3:5] / anchors)
    )  # width, height coordinates

    box_loss = mse(predictions_temp[..., 1:5][obj], target_temp[..., 1:5][obj])
    #print(predictions.shape,predictions_temp.shape,target.shape,target.shape, box_loss)
    # ================== #
    #   FOR CLASS LOSS   #
    # ================== #
    #print("class_loss",predictions[..., 5:][obj].shape, target[..., 5][obj])
    class_loss = entropy(
        (target[..., 5][obj]),
        (predictions[..., 5:][obj])
    )

    #print("__________________________________")
    #print("box loss",float(box_loss))
    #print("object loss",float(object_loss))
    #print("no object loss",float(no_object_loss))
    #print("class loss",float(class_loss))
    #print("\n")

    return (
        lambda_box * box_loss
        + lambda_obj * object_loss
        + lambda_noobj * no_object_loss
        + lambda_class * class_loss
    )

def yolo_loss(y_pred, y_target):
    scaling_S=tf.reshape(tf.repeat(config.S,[6,6,6]),[3,3,2])
    scaled_anchors = (
        tf.constant(config.ANCHORS)
        * scaling_S
    )
    loss = (myYoloLoss(y_pred[0], y_target[0], scaled_anchors[0])
            + myYoloLoss(y_pred[1], y_target[1], scaled_anchors[1])
            + myYoloLoss(y_pred[2], y_target[2], scaled_anchors[2])
            )

