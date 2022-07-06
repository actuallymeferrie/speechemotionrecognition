
!pip install hpelm

import hpelm

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


NUM_CLASS  =  7
ELM_HIDDEN_NEURONS  =  20


dscnn_model = tf.keras.models.load_model('gdrive/MyDrive/Models/FinalModel500Epochs.h5')

dir_list = 'Augmented Spectrogram'

batch_size = 5
img_height = 256
img_width = 256

train_ds = tf.keras.utils.image_dataset_from_directory(
  dir_list,
  shuffle="True",
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dir_list,
  shuffle="True",
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)



for image, label in train_ds.take(1):
  train_image_batch = image
  train_labels_batch = label

for image, label in val_ds.take(1):
  test_image_batch = image
  test_labels_batch = label

dscnn_model.summary()

layer_name = 'flatten' #name of the last layer

hidden_layer_model = Model(inputs = dscnn_model.input , outputs = dscnn_model.get_layer(layer_name).output )

# dscnn_train_result = hidden_layer_model.predict(train_image_batch)

dscnn_train_result = hidden_layer_model.predict(train_image_batch)

"""**TRAIN/FIT ELM MODEL**"""

train_label = np.expand_dims(train_labels_batch, -1)

target_train_oh = np_utils.to_categorical(train_labels_batch, NUM_CLASS)

elm_model = hpelm.elm.ELM(dscnn_train_result.shape[1] , NUM_CLASS)

elm_model.add_neurons( ELM_HIDDEN_NEURONS , func = 'sigm')

elm_model.train(dscnn_train_result, target_train_oh, 'ml')

"""**TEST ELM MODEL**"""

test_labels = np.expand_dims(test_labels_batch, -1)
target_test_oh = np_utils.to_categorical(test_labels_batch, NUM_CLASS)

#ELM model evaluation
dscnn_result = hidden_layer_model.predict(test_image_batch)
elm_result = elm_model.predict(dscnn_result)

elm_result_class = np.array([np.argmax(r) for r in elm_result])

confusion = elm_model.confusion(target_test_oh, elm_result)



# plt.figure(figsize=(10, 6))
# sns.heatmap(confusion, annot=True, 
#             fmt='g',
#             xticklabels=val_ds.class_names,
#             yticklabels=val_ds.class_names
#             )
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Class')
# plt.ylabel('Actual Class')
# plt.show()

from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, support = precision_recall_fscore_support(test_labels, elm_result_class)

# print(precision)
# print(recall)
# print(fscore)
# print(support)

# from sklearn.metrics import accuracy_score

# print('Accuracy: ', accuracy_score(test_labels, elm_result_class))

# # classification report
# from sklearn.metrics import classification_report
# print(classification_report(test_labels, elm_result_class))


def predict_one(test_image_batch):
  
  dscnn_elm_result = hidden_layer_model.predict(test_image_batch)
  predictions = elm_model.predict(dscnn_elm_result)

  print(np.argmax(predictions[55]))
  print(val_ds.class_names)

  
  plt.figure(figsize=(10, 10))
  for images, labels in val_ds.take(1):
      plt.imshow(images[55].numpy().astype("uint8"))
      plt.title(val_ds.class_names[labels[55]])
      plt.axis("off")