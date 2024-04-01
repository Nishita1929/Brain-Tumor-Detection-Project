from flask import Flask, render_template, request, redirect
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask import flash
from flask import session
import numpy as np
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras import backend as K
import os
import glob
import shutil

from keras.applications.vgg19 import VGG19

import numpy as np
#import sklearn
#import Keras packages

#tf.compat.v1.disable_eager_execution()


src_dir = "static/img"
dst_dir = "static/img1"

app = Flask(__name__)
app.secret_key = "super secret key"

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route('/', methods=['GET', 'POST'])
def upload():
    flash('')
    if request.method == 'POST' and 'photo' in request.files:
        	filename = photos.save(request.files['photo'])
        	for jpgfile in glob.iglob(os.path.join(src_dir, "*.*")):
        	  shutil.copy(jpgfile, dst_dir)
        
        	#vgg = VGG19(weights=None, input_shape=(256,256,3), include_top=False)
        	classifier = Sequential()

        	classifier.add(Convolution2D(64, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
        	classifier.add(MaxPooling2D(pool_size = (2, 2)))
        	classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
        	classifier.add(MaxPooling2D(pool_size = (2, 2)))
        	classifier.add(Convolution2D(16, (3, 3), activation = 'relu'))
        	classifier.add(MaxPooling2D(pool_size = (2, 2)))



        	classifier.add(Flatten())

	        #hidden layer
        	classifier.add(Dense(units= 256,activation = 'relu'))
        	classifier.add(Dropout(rate = 0.5))

        	#output layer
        	classifier.add(Dense(units = 4, activation = 'softmax'))

        	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        	classifier.summary()

         
        	print("Done1")
        	model=classifier
        
        
        
        	#loading saved weights
        	classifier.load_weights("BrainTumor.h5")


        	from keras.preprocessing.image import ImageDataGenerator
        
        	test_datagen = ImageDataGenerator(rescale=1./255)
        
        
        	itr1 = test_set1 = test_datagen.flow_from_directory(
                'static',
                target_size=(256, 256),
                batch_size=377,
                class_mode='categorical')
        
        	X1, y1 = itr1.next()
        	arr = classifier.predict(X1, batch_size=377, verbose=1)
        	#flash(arr)
        	arr = np.argmax(arr, axis=1)

        	"""
        	i = 0
        	j = 0
        	while(i < len(arr)):
        	  if(arr[i] == 1):
        	    j += 1
        	  i += 1
        	"""
        #flash('Images with no alcohol content found: ' + j)
        
        #for layer in classifier.layers:
        #    g=layer.get_config()
        #    h=layer.get_weights()
        #    print (g)
        #    print (h)
        
        #scores = classifier.evaluate_generator(test_set,62/32)
        	flash(str(arr[0]))
        	if(arr[0] == 0):
        	  flash('Image is of glioma')
        	elif(arr[0] == 1):
        	  flash('Image is of meningioma')
        	elif(arr[0] ==2):
        	  flash('Image is of notumor')
        	else:
        	  flash('Image is of pitutary')
        
        	#K.clear_session()
        
        	K.clear_session()
        	shutil.copyfile('static/img/' + filename,'static/img1/' + filename)
		
        
        	os.remove('static/img/' + filename)
        	return render_template('image.html', user_image = 'static/img1/' + filename)
    return render_template('image.html')


if __name__ == '__main__':
    app.run(debug=True)