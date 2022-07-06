from setup import *
NUM_CLASS  =  7
ELM_HIDDEN_NEURONS  =  20
# audio_file = open('YAF_back_angry.wav', 'rb')
audio_file = 'YAF_back_angry.wav'
# audio_bytes = audio_file.read()
# st.audio(audio_bytes, format='audio/wav')
data_visual(audio_file, 1)

dscnn_model = tf.keras.models.load_model('Models/500e.h5')
layer_name = 'flatten'

input = dscnn_model.input
output = dscnn_model.get_layer(layer_name).output

st.write("Input",input)
st.write("Output",output)
hidden_layer_model = Model(inputs = input  , outputs = output )
dscnn_train_result = hidden_layer_model.predict('mel_spectrogram_1.png')
#elm
# elm_model = hpelm.elm.ELM(dscnn_train_result.shape[1] , NUM_CLASS)
# elm_model.add_neurons( ELM_HIDDEN_NEURONS , func = 'sigm')

# dscnn_elm_result = hidden_layer_model.predict(test_image_batch)
# predictions = elm_model.predict(dscnn_elm_result)