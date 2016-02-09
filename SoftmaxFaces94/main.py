import os
import tensorflow as tf
import numpy as np
# from PIL import Image
# faces94_132_train faces94_132_test invalid
TRAIN_DIR = './faces94_132_train/'
TEST_DIR = './faces94_132_test/'
people_dict = {}
num_labels = 152
train_size = num_labels*16
test_size = num_labels*4
image_width = 180
image_height = 200
num_channels = 3 
image_pixels = image_width * image_height * num_channels


def get_images_filenames_and_labels(directory):
  files_dir = os.listdir(directory)
  people_count = len(files_dir)
  all_people_data = []
  for i in range(people_count):
    person_dir = files_dir[i]
    person_images_files = os.listdir(directory + person_dir + '/')
    for person_file in person_images_files:
      all_people_data.append(directory + person_dir + '/' + person_file)
  return all_people_data

def image_reader_and_decoder(images_paths):
  reader = tf.WholeFileReader()
  if len(images_paths) > 0:
    jpeg_file_queue = tf.train.string_input_producer(images_paths)
    jkey, jvalue = reader.read(jpeg_file_queue)
    j_img = tf.image.decode_jpeg(jvalue)
#     reshaped_image = tf.reshape(j_img, [250, 250, 3])
    reshaped_image = tf.reshape(j_img, [200, 180, 3])
    reshaped_image = tf.cast(reshaped_image, tf.float32)
  return (reshaped_image, jkey)

def get_input_as_np(train=True):
  if train :
    directory = TRAIN_DIR
  else:
    directory = TEST_DIR
  images_paths = get_images_filenames_and_labels(directory) 
  j_img, jkey = image_reader_and_decoder(images_paths)
  images = []
  labels = []
  init_op = tf.initialize_all_variables()
  unique_index = 0
  with tf.Session() as sess:
    sess.run(init_op)
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      # print('Trial x')
      for i in range(len(images_paths)):
        jpg, k = sess.run([j_img, jkey])
        person_name = k.split('/')[-2]
#         Image.fromarray(np.asarray(jpg.astype('uint8'))).show()
        if people_dict.has_key(person_name):
          label = people_dict[person_name]
        else:
          people_dict[person_name] = unique_index
          label = unique_index
          unique_index = unique_index + 1
        images.append(jpg)
        labels.append(label)
      
    except Exception as e:
      print(e.message)
    finally:
      # When done, ask the threads to stop.
      # print('Request stop x')
      coord.request_stop()
    coord.join(threads)
  return np.asarray(images).astype('float32'), np.asarray(labels).astype('int32')

def convert_to_onehot(labels):  
  onehot = np.zeros((labels.shape[0], num_labels))
  onehot[np.arange(labels.shape[0]), labels] = 1
  return onehot

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])
  
# GRAPH
graph = tf.Graph()
with graph.as_default():
  np_images, np_labels = get_input_as_np(train=True)
  np_images_reshaped = np_images.reshape((train_size, image_pixels))
  np_labels = convert_to_onehot(np_labels)
  np_images_test, np_labels_test = get_input_as_np(train=False)
  np_images_test_reshaped = np_images_test.reshape((test_size, image_pixels))
  np_labels_test = convert_to_onehot(np_labels_test)
  
  tf_train_dataset = tf.placeholder(tf.float32, shape=(train_size, image_pixels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(train_size, num_labels))
  # tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(np_images_test_reshaped)
  
  # Variables.
  softmax_weights = tf.Variable(tf.zeros([image_pixels, num_labels]))
  # tf.Variable( tf.truncated_normal([image_pixels, num_labels]) )
  softmax_biases = tf.Variable( tf.zeros([num_labels]) )
  
  # Training computation.
  logits = tf.matmul(tf_train_dataset, softmax_weights) + softmax_biases
  loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) )
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(1e-6).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, softmax_weights) + softmax_biases)

# SESSION
num_steps = 101
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    feed_dict = {tf_train_dataset : np_images_reshaped, tf_train_labels : np_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1 == 0):
      print("Train loss at step %d: %f" % (step, l))
      print("Train accuracy: %.1f%%" % accuracy(predictions, np_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), np_labels_test))