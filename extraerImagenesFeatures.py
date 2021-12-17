# Normalization:
#Zero centering means that you process your image so that the mean of your image lies on the zero. Mathematically this can be done by calculating the mean in your images and subtracting each image item with that mean.
#The mean and standard deviation required to standardize pixel values can be calculated from the pixel values in each image only (sample-wise) or across the entire training dataset (feature-wise).
#The inputs pixel values are scaled between -1 and 1, sample-wise. 
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Ignoramos los mensajes de aviso de CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from matplotlib import pyplot

import math

model = InceptionV3(include_top=False, input_shape=(299, 299, 3))
model.summary()
"""
# Extraemos las capas. Cogemos una de cada 10
contador = 0
layer_outputs = []
layer_names = []
for layer in model.layers:
    if contador % 10 == 0:
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)        
    contador +=1

#print(str(contador))
#layer_outputs = [layer.output for layer in model.layers[:12]] 

activation_model = Model(inputs=model.input, outputs=layer_outputs)
"""


img_path='ejemplo_original.png'
# Define a new Model, Input= image 
# Output= intermediate representations for all layers in the  
# previous model after the first.
# Como Inception tiene muchas capas, cogemos únicamente 1 de cada 10

successive_outputs = []
layer_names = []

contador = 0
for layer in model.layers: 
    if (contador % 60):
        successive_outputs.append(layer.output)
        layer_names.append(layer.name)
    contador += 1

# successive_outputs = [layer.output for layer in model.layers[1:]]#visualization_model = Model(img_input, successive_outputs)
visualization_model = Model(inputs = model.input, outputs = successive_outputs)#Load the input image
img = load_img(img_path, target_size=(299, 299))# Convert ht image to Array of dimension (150,150,3)
x = img_to_array(img)
x = preprocess_input(x)
save_img('normalizada.png', x)

x   = x.reshape((1,) + x.shape)# Rescale by 1/255
x /= 255.0# Let's run input image through our vislauization network
# to obtain all intermediate representations for the image.
successive_feature_maps = visualization_model.predict(x)# Retrieve are the names of the layers, so can have them as part of our plot
# layer_names = [layer.name for layer in model.layers]
contador = 0
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    print(feature_map.shape)
    if len(feature_map.shape) == 4:
        # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
   
        n_features = feature_map.shape[-1]  # number of features in the feature map
        size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
        # We will tile our images in this matrix
        # display_grid = np.zeros((size, size * n_features))
        
        h = math.sqrt(float(n_features))
        h = int(h)
        v = h

        # Limitamos a 8 por representación
        if h > 8:
            h = 8
            v = 8        

        display_grid2 = np.zeros((size * h, size * v))
    
        # Postprocess the feature to be visually palatable
        ch = 0
        cv = 0
        for i in range(n_features):
            x  = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std ()
            x *=  64
            x += 128
            x  = np.clip(x, 0, 255).astype('uint8')
            # Tile each filter into a horizontal grid
            #display_grid[:, i * size : (i + 1) * size] = x# Display the grid
            display_grid2[ch * size : size * (ch + 1), cv * size : size * (cv + 1)] = x # Display the grid
            cv += 1
            if cv == v:
                ch += 1
                cv = 0
            if ch == h:
                break            
            
        scale = 20. / n_features
        plt.figure( figsize=(scale * n_features, scale) )
        plt.title ( layer_name )
        plt.grid  ( False )
        #plt.imshow( display_grid, aspect='auto', cmap='viridis' )
        #plt.imsave('paso' + str(contador) + '.png', display_grid, cmap='viridis')
        plt.imsave('paso_' + str(contador) + '_' + layer_name + '.png', display_grid2, cmap='viridis')
        contador += 1

"""
image = load_img('ejemplo.png', target_size=(299, 299))
image = img_to_array(image)
image = preprocess_input(image)
image = np.array([image])
#predictions = model.predict(input_arr)
#image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model




feature_maps = model.predict(image)



for feature_map in feature_maps:  
    if len(feature_map.shape) == 4:
        k = feature_map.shape[-1]  
        size=feature_map.shape[1]
        image_belt = 
        for i in range(k):
            feature_image = feature_map[0, :, :, i]
            feature_image-= feature_image.mean()
            feature_image/= feature_image.std ()
            feature_image*=  64
            feature_image+= 128
            feature_image= np.clip(x, 0, 255).astype('uint8')
            image_belt[:, i * size : (i + 1) * size] = feature_image   

   scale = 20. / k
    plt.figure( figsize=(scale * k, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( image_belt, aspect='auto')



print(str(feature_maps.shape))
# plot the output from each block
square = 8
contador = 0
for fmap in feature_maps:
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            print(str(fmap.shape))
            pyplot.imshow(fmap[ix-1, :, :], cmap='gray')
            #pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()
    pyplot.savefig('paso' + str(contador) + '.png')
    contador +=1
"""
"""activations = activation_model.predict(image) 
# Returns a list of five Numpy arrays: one array per layer activation

images_per_row = 5

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


plt.savefig('pasoG.png')"""

"""first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')
plt.savefig('paso1.png')

first_layer_activation = activations[6]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')
plt.savefig('paso6.png')
"""
