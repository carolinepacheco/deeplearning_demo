import numpy as np
import argparse
import cv2
import os
from keras.preprocessing import image as image_utils
from keras.optimizers import SGD
from models.vgg16 import VGG16
from models.imagenet_utils import decode_predictions
from models.imagenet_utils import preprocess_input

# python dl_image_prediction_folder.py -i input_dir -o output_dir

def main():
  parser = argparse.ArgumentParser(description='Classify images')
  parser.add_argument('-i','--img_in', help='Input folder', required=True)
  parser.add_argument('-o','--img_out', help='Output folder', required=True)
  
  args = parser.parse_args()
  if args.img_in:
    print("reading from: %s..." % args.img_in)
  if args.img_out:
    print("output: %s..." % args.img_out)
  
  img_in = args.img_in
  img_out = args.img_out
  classify(img_in, img_out)

def classify(img_in, img_out):
    images = list()
    
    # load the VGG16 network
    print("[INFO] loading network...")
    model = VGG16(weights="imagenet")
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy') 
    
    for item in os.listdir(img_in):
        if not item.startswith('.') and os.path.isfile(os.path.join(img_in, item)):
            print (item)
            img = cv2.imread(os.path.join(img_in, item))
            
            if img is not None:
                 images.append(img)
                 print("[INFO] loading and preprocessing image...")
                 image = cv2.resize(img, (224, 224)) 
                 image = image_utils.img_to_array(image)
                 # our image is now represented by a NumPy array of shape (3, 224, 224),
                 # but we need to expand the dimensions to be (1, 3, 224, 224) so we can
                 # pass it through the network -- we'll also preprocess the image by
                 # subtracting the mean RGB pixel intensity from the ImageNet dataset
                 image = np.expand_dims(image, axis=0)
                 image = preprocess_input(image)
                
                 # classify the image
                 print("[INFO] classifying image...")
                 preds = model.predict(image)
                 P = decode_predictions(preds)
                 (imagenetID, label, prob) = P[0][0]
                 #print('Predicted:', decode_predictions(preds,1))
                 print(decode_predictions(preds,3))

                 # display the predictions to our screen
                 print("ImageNet ID: {}, Output label: {},  Prob: {}".format(imagenetID, label, round(prob,4)))
                 cv2.putText(img, "Output label: {}".format(label), (10, 30), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (102, 0, 255), 2)
                 cv2.putText(img, "Prob: {0:.0f}%".format(prob * 100), (10, 90), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                 #create a new folder
                 foldoutcheck = os.path.exists(img_out)
                 if foldoutcheck == False:
                     os.makedirs(img_out)
                 foldcheck = os.path.exists(img_out+"/"+label)
                 if foldcheck == True:
                     cv2.imwrite(img_out+"/"+label+"/"  + item, img)
                 else:
                     os.makedirs(img_out+"/"+label)
                     cv2.imwrite(img_out+"/"+label+"/"  + item, img)
                                   
if __name__ == "__main__":
  main()
