from setup import *

audio_file = open('YAF_back_angry.wav', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/wav')

dscnn_model = tf.keras.models.load_model('Models/500e.h5')
layer_name = 'flatten'

input = dscnn_model.input
output = dscnn_model.get_layer(layer_name).output

hidden_layer_model = Model(inputs = input  , outputs = output )