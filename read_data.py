import numpy
import logging
from sklearn.datasets import fetch_lfw_people

n_classes = 0

def extract_images():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    lfw_people = fetch_lfw_people(data_home= '~/Desktop/FooFoo', download_if_missing=True, 
      resize= 1, slice_= (slice(25, 225, None), slice(25, 225, None)), color=False, min_faces_per_person=5)

    target_names = lfw_people.target_names
    global n_classes
    n_classes = target_names.shape[0]
    lfw_people_images = lfw_people.images
    lfw_people_images_resized = numpy.resize(lfw_people_images,(lfw_people_images.shape[0], 40000,))
    lfw_people_target_ids = lfw_people.target
    return (lfw_people_images_resized, lfw_people_target_ids)

def read_data_sets():
  images, labels = extract_images()
  print(images.shape)

  train_images = images[:4500]
  train_labels = labels[:4500]

  test_images = images[4500:5500]
  test_labels = labels[4500:5500]

  validation_images = images[5500:5900]
  validation_labels = labels[5500:5900]

  training_data = (train_images, train_labels)
  validation_data = (validation_images, validation_labels)
  test_data = (test_images, test_labels)

  return (training_data, validation_data, test_data)
