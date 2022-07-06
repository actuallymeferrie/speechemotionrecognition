from setup import *
from keras.models import load_model, Sequential, Model
NUM_CLASS  =  7
ELM_HIDDEN_NEURONS  =  20


dscnn_model = tf.keras.models.load_model('Models/500e.h5')
layer_name = 'flatten' #name of the last layer
hidden_layer_model = Model(inputs = dscnn_model.input , outputs = dscnn_model.get_layer(layer_name).output )

file_audio = 'YAF_back_angry.wav'
plt.savefig('mel_spectrogram_1.png',transparent=True,bbox_inches='tight',pad_inches=0, dpi=72)
 

dscnn_train_result = hidden_layer_model.predict('mel_spectrogram_1.png')
st.write("DSCNN TRAIN RESULT",dscnn_train_result)