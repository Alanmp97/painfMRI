from util_fcn import *
from tensorflow.keras import backend as k
from keras.callbacks import ModelCheckpoint

params = DefineParameters(batch_train=10, epochs=1, batch_test=42, axis_select=0)
"""
train_dataset = tf.data.Dataset.from_generator(lambda: train_data_generator_func(params),
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=((None, 128, 128, 22), (None,)))

test_dataset = tf.data.Dataset.from_generator(lambda: test_data_generator(params),
                                              output_types=tf.float32, output_shapes=(None, 128, 128, 22))
"""
#para entrenar con una sola sesion 
train_dataset = tf.data.Dataset.from_generator(lambda: just_one_subj(params),
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=((None, 128, 128, 22), (None,)))

test_dataset = tf.data.Dataset.from_generator(lambda: just_one_subj(params, train = False),
                                              output_types=tf.float32, output_shapes=(None, 128, 128, 22))


# Generate a print
print('==========================================================================')
print(f'Starting', params.axial_coords[params.axis_select], 'model training ... ...')

# build the model and compile
model = build_CNN_model(params)

# create callback
filepath = './Models/cnn/' + params.axial_coords[params.axis_select] + '_resnet.epoch{epoch:02d}-loss{' \
                                                                       'val_loss:.2f}.hdf5 '
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

# Fit data to model
model.fit(train_dataset, epochs=params.epochs, steps_per_epoch=params.steps_per_epoch, validation_data=test_dataset,
          verbose=0, callbacks=callbacks)

# Generate prediction
print('==========================================================================')
print(f'Starting', params.axial_coords[params.axis_select], 'model testing ... ...')

prediction = model.predict(test_dataset, steps=1)
    
print(f'Results:')
print(params.axial_coords[params.axis_select], f'prediction:')
print(prediction)
print(params.axial_coords[params.axis_select], f'target:')
print(load_test_targets(params))
print('==========================================================================')

# Save model
model.save('./Models/cnn/cnn_' + params.axial_coords[params.axis_select], save_format='h5')

# Cleared tensorflow.keras session
k.clear_session()

print('FIN')
