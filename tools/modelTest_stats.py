import tensorflow as tf 
from model import x_test, y_test

model = tf.keras.models.load_model("lstm.h5") 
 



loss, accuracy = model.evaluate(x_test, y_test) 
print(loss, accuracy)