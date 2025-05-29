from keras.models import load_model
import tensorflow as tf

def load_keras_model():
    model = load_model('model.h5')  # or whatever your saved Keras model path is
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = load_keras_model()
tf.keras.models.save_model(model, 'model.h5')



