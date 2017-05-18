from keras.models import load_model
import numpy

#load the model
model = load_model('my_model_csv.h5')

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# evaluate the model
scores = model.evaluate(X, Y)
print("Accuracy: \n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))