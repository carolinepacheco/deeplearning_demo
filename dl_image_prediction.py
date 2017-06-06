# import the necessary packages
import numpy as np
import argparse
import cv2
from keras.preprocessing import image as image_utils
from keras.optimizers import SGD
from keras.preprocessing import image
from models.vgg16 import VGG16
from models.imagenet_utils import decode_predictions
from models.imagenet_utils import preprocess_input
from models.imagenet_utils import preprocess_input, decode_predictions

# python dl_image_prediction.py -i images/bear.jpg -o output/bear.jpg

def main():
  parser = argparse.ArgumentParser(description='Classify images')
  parser.add_argument('-i','--img_in', help='Input image', required=True)
  parser.add_argument('-o','--img_out', help='Output image', required=True)
  
  args = parser.parse_args()
  if args.img_in:
    print("reading %s..." % args.img_in)
  if args.img_out:
    print("output file %s..." % args.img_out)
  
  img_in = args.img_in
  img_out = args.img_out
  classify(img_in, img_out)


def classify(img_in, img_out):
  # load image   
  orig = cv2.imread(img_in)
  
  # load the input image using the Keras helper utility while ensuring
  # that the image is resized to 224x224 pxiels, the required input
  # dimensions for the network -- then convert the PIL image to a
  # NumPy array
  print("[INFO] loading and preprocessing image...")
  image = image_utils.load_img(img_in, target_size=(224, 224))
  image = image_utils.img_to_array(image)

  # our image is now represented by a NumPy array of shape (3, 224, 224),
  # but we need to expand the dimensions to be (1, 3, 224, 224) so we can
  # pass it through the network -- we'll also preprocess the image by
  # subtracting the mean RGB pixel intensity from the ImageNet dataset
  image = np.expand_dims(image, axis=0)
  image = preprocess_input(image)

  # load the VGG16 network
  print("[INFO] loading network...")
  model = VGG16(weights="imagenet")
  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy')
   
  # classify the image
  print("[INFO] classifying image...")
  preds = model.predict(image)
  P = decode_predictions(preds)
  (imagenetID, label, prob) = P[0][0]
  #print('Predicted:', decode_predictions(preds,1))
  print(decode_predictions(preds,3))

  # display the predictions to our screen
  print("ImageNet ID: {}, Output label: {},  Prob: {}".format(imagenetID, label, round(prob,4)))
  cv2.putText(orig, "Output label: {}".format(label), (10, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (102, 0, 255), 2)
  cv2.putText(orig, "Prob: {0:.0f}%".format(prob * 100), (10, 90), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
  # cv2.imshow(label, orig)
  # cv2.waitKey(2500)

  cv2.imwrite(img_out, orig)


if __name__ == "__main__":
  main()
