from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from das_lib.funs import train_test_data_recovery
from das_lib.evaluation import get_confusion_matrix
# def cnn1d_1(input_size, n_outputs):
#     inputs = keras.Input(shape=input_size)
#     conv1d_1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_size)
#     x = conv1d_1(inputs)
#     conv1d_2 = layers.Conv1D(filters=64, kernel_size=3, activation='relu')
#     x = conv1d_2(x)
#     dropout = layers.Dropout(0.5)
#     x = dropout(x)
#     maxpooling1d = keras.layers.MaxPooling1D(pool_size=2)
#     x = maxpooling1d(x)
#     flatten = layers.Flatten()
#     x = flatten(x)
#     dense_1 = layers.Dense(100, activation='relu')
#     x = dense_1(x)
#     dense_2 = layers.Dense(n_outputs, activation='softmax')
#     outputs = dense_2(x)
#     model = keras.Model(inputs=inputs, outputs=outputs, name="1dcnn_model")
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

def cnn1d(input_size, n_outputs):
    inputs = keras.Input(shape=input_size)
    conv1d_1 = layers.Conv1D(filters=16, kernel_size=64, strides=2, activation='relu', input_shape=input_size)
    x = conv1d_1(inputs)
    maxpooling1d_1 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
    x = maxpooling1d_1(x)
    conv1d_2 = layers.Conv1D(filters=32, kernel_size=32, strides=2, activation='relu')
    x = conv1d_2(x)
    maxpooling1d_2 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
    x = maxpooling1d_2(x)  
    conv1d_3 = layers.Conv1D(filters=16, kernel_size=16, strides=2, activation='relu')
    x = conv1d_3(x)
    flatten = layers.Flatten()
    x = flatten(x)  
    dense_1 = layers.Dense(128, activation='relu')
    x = dense_1(x)
    dense_2 = layers.Dense(64, activation='relu')
    x = dense_2(x)
    dense_3 = layers.Dense(n_outputs, activation='softmax')
    outputs = dense_3(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="1dcnn_model")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# def cnn1d_3(input_size, n_outputs):
#     inputs = keras.Input(shape=input_size)
#     conv1d_1 = layers.Conv1D(filters=16, kernel_size=64, strides=2, activation='relu', input_shape=input_size)
#     x = conv1d_1(inputs)
#     maxpooling1d_1 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
#     x = maxpooling1d_1(x)
#     conv1d_2 = layers.Conv1D(filters=32, kernel_size=32, strides=2, activation='relu')
#     x = conv1d_2(x)
#     flatten = layers.Flatten()
#     x = flatten(x)  
#     dense_1 = layers.Dense(128, activation='relu')
#     x = dense_1(x)
#     dense_2 = layers.Dense(64, activation='relu')
#     x = dense_2(x)
#     dense_3 = layers.Dense(n_outputs, activation='softmax')
#     outputs = dense_3(x)
#     model = keras.Model(inputs=inputs, outputs=outputs, name="1dcnn_model")
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# def cnn1d_svm(input_size, n_outputs):
#     inputs = keras.Input(shape=input_size)
#     conv1d_1 = layers.Conv1D(filters=16, kernel_size=64, strides=2, activation='relu', input_shape=input_size)
#     x = conv1d_1(inputs)
#     maxpooling1d_1 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
#     x = maxpooling1d_1(x)
#     conv1d_2 = layers.Conv1D(filters=32, kernel_size=32, strides=2, activation='relu')
#     x = conv1d_2(x)
#     maxpooling1d_2 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
#     x = maxpooling1d_2(x)
#     conv1d_3 = layers.Conv1D(filters=16, kernel_size=12, strides=2, activation='relu')
#     x = conv1d_3(x)
#     flatten = layers.Flatten()
#     x = flatten(x)  
#     dense_1 = layers.Dense(128, activation='relu')
#     x = dense_1(x)
#     dense_2 = layers.Dense(64, activation='relu')
#     x = dense_2(x)
#     dense_3 = layers.Dense(n_outputs, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))
#     outputs = dense_3(x)
#     model = keras.Model(inputs=inputs, outputs=outputs, name="1dcnn_model")
#     model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
#     return model

# Training 
def train(model, traindata, testdata, repeats=1, epoch=10, batch_size=32, verbose=True):
    for r in range(repeats):
        model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, callbacks=[cp_callback], validation_data=(testX, testy), verbose=verbose)
        _, score = model.evaluate(testX,testy, batch_size=batch_size, verbose=verbose)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
        
        
# def cnn1d_svm_huber(input_size, n_outputs):
#     inputs = keras.Input(shape=input_size)
#     conv1d_1 = layers.Conv1D(filters=16, kernel_size=64, strides=2, activation='relu', input_shape=input_size)
#     x = conv1d_1(inputs)
#     maxpooling1d_1 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
#     x = maxpooling1d_1(x)
#     conv1d_2 = layers.Conv1D(filters=32, kernel_size=32, strides=2, activation='relu')
#     x = conv1d_2(x)
#     maxpooling1d_2 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
#     x = maxpooling1d_2(x)
#     conv1d_3 = layers.Conv1D(filters=16, kernel_size=12, strides=2, activation='relu')
#     x = conv1d_3(x)
#     flatten = layers.Flatten()
#     x = flatten(x)  
#     dense_1 = layers.Dense(128, activation='relu')
#     x = dense_1(x)
#     dense_2 = layers.Dense(64, activation='relu')
#     x = dense_2(x)
#     dense_3 = layers.Dense(n_outputs, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))
#     outputs = dense_3(x)
#     model = keras.Model(inputs=inputs, outputs=outputs, name="1dcnn_model")
#     model.compile(loss='Huber', optimizer='adam', metrics=['accuracy'])
#     return model


def my_loss_fn(y_true, y_pred): #y_true and y_pred all are confidence vectors 
    cce = tf.keras.losses.CategoricalCrossentropy()
    ce_loss = cce(y_true, y_pred).numpy() # numpy is a methods under cce object and return float, when using it in loss, model.compile needs to enable run eagerly 
    squared_difference = tf.square(y_true - y_pred) 
    return tf.reduce_mean(squared_difference, axis=-1)+ce_loss  # Note the `axis=-1` meeans the last dimension.  this term is MSE + CE


def cnn1d_ce(input_size, n_outputs):
    inputs = keras.Input(shape=input_size)
    conv1d_1 = layers.Conv1D(filters=16, kernel_size=64, strides=2, activation='relu', input_shape=input_size)
    x = conv1d_1(inputs)
    maxpooling1d_1 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
    x = maxpooling1d_1(x)
    conv1d_2 = layers.Conv1D(filters=32, kernel_size=32, strides=2, activation='relu')
    x = conv1d_2(x)
    maxpooling1d_2 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
    x = maxpooling1d_2(x)
    conv1d_3 = layers.Conv1D(filters=16, kernel_size=12, strides=2, activation='relu')
    x = conv1d_3(x)
    flatten = layers.Flatten()
    x = flatten(x)  
    dense_1 = layers.Dense(128, activation='relu')
    x = dense_1(x)
    dense_2 = layers.Dense(64, activation='relu')
    x = dense_2(x)
    dense_3 = layers.Dense(n_outputs, activation='softmax')
    outputs = dense_3(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="1dcnn_model")
#     model.compile(loss='categorical_hinge', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=my_loss_fn, optimizer='adam', metrics=['accuracy'], run_eagerly=True)
    return model


def print_average_cm_custom_loss(repeats,  models_root_folder, training_indices,class_num, data_dir, data_files, labels_dir, labels_files, ground_truth_unique_labels):
    ten_conf_matrixs = []
    for i in range(repeats):
        model_folder = models_root_folder + "model_{0}".format(i+1)
        training_index = training_indices.iloc[i, 1:]
        (x_train_new, y_train_new), (x_test_new, y_test_new)  = train_test_data_recovery(data_dir, data_files, labels_dir, labels_files, training_index)
        reconstructed_model = tf.keras.models.load_model(model_folder, custom_objects={'my_loss_fn': my_loss_fn})
        reconstructed_model.run_eagerly = True
        cm, sample_label = get_confusion_matrix(reconstructed_model, x_test_new, y_test_new, class_num, ground_truth_unique_labels)
        ten_conf_matrixs.append(cm)
    print('testing label: {0}'.format(sample_label))
    averaged_cm = sum(ten_conf_matrixs)/10
    print("average cm: {0}".format(averaged_cm))
    return averaged_cm

def print_average_cm_unseen_custom_loss(repeats,  models_root_folder, class_num, unseen_data, unseen_label, ground_truth_unique_labels):
    ten_conf_matrixs = []
    y0 = 1
    unseen_label = tf.keras.utils.to_categorical(unseen_label - y0, num_classes=class_num)
    for i in range(repeats):
        model_folder = models_root_folder + "model_{0}".format(i+1)
        reconstructed_model = tf.keras.models.load_model(model_folder, custom_objects={'my_loss_fn': my_loss_fn})
        reconstructed_model.run_eagerly = True
        cm, sample_label = get_confusion_matrix(reconstructed_model, unseen_data, unseen_label, class_num, ground_truth_unique_labels)
        ten_conf_matrixs.append(cm)
    print('testing label: {0}'.format(sample_label))
    averaged_cm = sum(ten_conf_matrixs)/10
    print("average cm: {0}".format(averaged_cm))
    return averaged_cm


def cnn1d_dropout(input_size, n_outputs):
    inputs = keras.Input(shape=input_size)
    conv1d_1 = layers.Conv1D(filters=16, kernel_size=64, strides=2, activation='relu', input_shape=input_size)
    x = conv1d_1(inputs)
    maxpooling1d_1 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
    x = maxpooling1d_1(x)
    d_1 = layers.Dropout(0.5)
    x = d_1(x)
    conv1d_2 = layers.Conv1D(filters=32, kernel_size=32, strides=2, activation='relu')
    x = conv1d_2(x)
    maxpooling1d_2 = keras.layers.MaxPooling1D(pool_size=8, strides=8)
    x = maxpooling1d_2(x)
    d_2 = layers.Dropout(0.5)
    x = d_2(x)
    conv1d_3 = layers.Conv1D(filters=16, kernel_size=12, strides=2, activation='relu')
    x = conv1d_3(x)
    maxpooling1d_2 = keras.layers.MaxPooling1D(pool_size=5, strides=8)
    x = maxpooling1d_2(x)    
    d_3 = layers.Dropout(0.5)
    x = d_3(x)
    flatten = layers.Flatten()
    x = flatten(x)  
    dense_1 = layers.Dense(128, activation='relu')
    x = dense_1(x)
    dense_2 = layers.Dense(64, activation='relu')
    x = dense_2(x)
    dense_3 = layers.Dense(n_outputs, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))
    outputs = dense_3(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="1dcnn_model")
    model.compile(loss="huber", optimizer='adam', metrics=['accuracy'])
    return model


def cnn2d_dropout(input_size, n_outputs):
    inputs = keras.Input(shape=input_size)
    conv1d_1 = layers.Conv2D(16, (3, 64), activation='relu')
    x = conv1d_1(inputs)
    maxpooling2d_1 = keras.layers.MaxPooling2D(pool_size=(1,8), strides=(4,8))
    x = maxpooling2d_1(x)
    d_1 = layers.Dropout(0.5)
    x = d_1(x)
    conv1d_2 = layers.Conv2D(32, (1, 16), activation='relu')
    x = conv1d_2(x)
    maxpooling2d_2 = keras.layers.MaxPooling2D(pool_size=(1,8), strides=(2,8))
    x = maxpooling2d_2(x)
    d_2 = layers.Dropout(0.5)
    x = d_2(x)  
    conv1d_3 = layers.Conv2D(16, (1,8),  activation='relu')
    x = conv1d_3(x)
    maxpooling2d_3 = keras.layers.MaxPooling2D(pool_size=(1,8), strides=(1,8))
    x = maxpooling2d_3(x)
    d_3 = layers.Dropout(0.5)
    x = d_3(x)     
    flatten = layers.Flatten()
    x = flatten(x)
    dense_1 = layers.Dense(128, activation='relu')
    x = dense_1(x)
    dense_2 = layers.Dense(64, activation='relu')
    x = dense_2(x)
    dense_3 = layers.Dense(n_outputs, kernel_regularizer=tf.keras.regularizers.L2(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01))
    outputs = dense_3(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="1dcnn_model")
    model.compile(loss="huber", optimizer='adam', metrics=['accuracy'])
    return model
