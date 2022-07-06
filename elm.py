from setup import *

NUM_CLASS  =  7
ELM_HIDDEN_NEURONS  =  20


dscnn_model = tf.keras.models.load_model('Models/500e.h5')
layer_name = 'flatten' #name of the last layer
hidden_layer_model = Model(inputs = dscnn_model.input , outputs = dscnn_model.get_layer(layer_name).output )
dscnn_train_result = hidden_layer_model.predict(train_image_batch)
st.write("DSCNN TRAIN RESULT",dscnn_train_result)
train_label = np.expand_dims(train_labels_batch, -1)
target_train_oh = np_utils.to_categorical(train_labels_batch, NUM_CLASS)
elm_model = hpelm.elm.ELM(dscnn_train_result.shape[1] , NUM_CLASS)
elm_model.add_neurons( ELM_HIDDEN_NEURONS , func = 'sigm')
elm_model.train(dscnn_train_result, target_train_oh, 'ml')
test_labels = np.expand_dims(test_labels_batch, -1)
target_test_oh = np_utils.to_categorical(test_labels_batch, NUM_CLASS)
dscnn_result = hidden_layer_model.predict(test_image_batch)
elm_result = elm_model.predict(dscnn_result)
elm_result_class = np.array([np.argmax(r) for r in elm_result])
confusion = elm_model.confusion(target_test_oh, elm_result)
st.write(predict_one(test_image_batch))