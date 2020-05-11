#!/usr/bin/env python
# coding: utf-8
#importing modules
import tensorflow as tf
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import dlib
images = []
labels = []
#function to detect faces using dlib
'''def detect_faces(image):
    face_detector = dlib.get_frontal_face_detector()
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]
    return face_frames
'''
#defining master directory
master_dir = ""
for dir_ in os.listdir( master_dir ):
    for imagename in os.listdir( os.path.join( master_dir , dir_ ) ):
        print( imagename )
        path = os.path.join( master_dir , dir_ , imagename )
        image = np.asarray( Image.open( path ) )
        detected_faces = image
        if len( detected_faces ) is not 0:
            face = Image.fromarray( image ).crop( detected_faces[0] ).resize((128,128))
            images.append( np.asarray( face ) / 255 )
            labels.append( 0 if dir_ == 'with_mask' else 1 )
        


# In[ ]:


#np.save('/home/jarvis/Desktop/ShubhamWork/npyfiles/mask-nonmask-images.npy', images)
#np.save('/home/jarvis/Desktop/ShubhamWork/npyfiles/mask-nonmask-labels.npy', labels)


# In[ ]:



#images = np.load( '/home/jarvis/Desktop/ShubhamWork/npyfiles/mask-nonmask-images.npy' ) 
#labels = np.load( '/home/jarvis/Desktop/ShubhamWork/npyfiles/mask-nonmask-images.npy' )


# In[ ]:



X = np.array( images )
Y = tf.keras.utils.to_categorical( labels , num_classes=2 )

train_features, test_features, train_labels, test_labels = train_test_split( X , Y ,test_size=0.2)


# In[ ]:


print( train_features.shape )
print( train_labels.shape )


# In[ ]:


# In[ ]:



rate = 0.4

mobilenet = tf.keras.applications.MobileNetV2( include_top=False , input_shape=(64,64,3 ), pooling='avg' , weights='imagenet' )
x = tf.keras.layers.Dense( 512 , activation='relu' )( mobilenet.output )
x = tf.keras.layers.Dropout( rate )(x)
x = tf.keras.layers.Dense( 256 , activation='relu' )(x)
x = tf.keras.layers.Dropout( rate )(x)
x = tf.keras.layers.Dense( 128 , activation='relu' )(x)
x = tf.keras.layers.Dropout( rate )(x)
x = tf.keras.layers.Dense( 64 , activation='relu' )(x)
x = tf.keras.layers.Dropout( rate )(x)
x = tf.keras.layers.Dense( 32 , activation='relu' )(x)
x = tf.keras.layers.Dropout( rate )(x)
x = tf.keras.layers.Dense( 16 , activation='relu' )(x)
x = tf.keras.layers.Dropout( rate )(x)
outputs = tf.keras.layers.Dense( 2 , activation='softmax' )(x)

model = tf.keras.Model( mobilenet.input , outputs )
model.summary()
 


# In[ ]:



model.compile( loss='categorical_crossentropy' , optimizer=tf.keras.optimizers.Adam( learning_rate=0.0001 ) , metrics=[ 'acc' ] )
model.fit( train_features , train_labels , validation_data=( test_features , test_labels ) , epochs=100 , batch_size=1 ) 


# In[ ]:



model.evaluate( test_features , test_labels )


# In[ ]:



converter = tf.lite.TFLiteConverter.from_keras_model( model )
converter.post_training_quantization = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
buffer = converter.convert()
open( 'Classifier.tflite' , 'wb' ).write( buffer )

