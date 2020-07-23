import augmentors as augmentors
import data_loader as data
import resnet as resnet_model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm.notebook import tqdm
from IPython.display import clear_output
import threading

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.datasets import *
from tensorflow.keras.applications import *
from tensorflow.keras.losses import *
from tensorflow.keras.experimental import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

references = open("simclr_storage_queue/image_urls_decoded.txt", "r").read().split("\n")
print(len(references), "images")

image_shape = (224, 224, 3)

batch_size = 32
max_queue_size = batch_size * 5
queue_fill_rate = 2

images_per_thread = 16
max_threads = 4
batch_downloader = data.BatchDownloader(batch_size * queue_fill_rate, images_per_thread, max_threads)

sk_ratio = 0.0625
input_shape = (224, 224, 3)
width_multiplier = 1
resnet_depth = 50
global_bn = True
batch_norm_decay = 0.9
train_mode = "pretrain"
se_ratio = 0
hidden_layer_sizes = [256, 128, 50]

tf.compat.v1.reset_default_graph()

# base_model = ResNet50(input_shape = input_shape, weights = None)
base_model = resnet_model.resnet_v1(resnet_depth, width_multiplier)
base_model = base_model(Input(shape = input_shape), is_training = True)
base_model.trainable = True

inputs = Input(shape = image_shape)

h = base_model(inputs, training = True)
h = Reshape((1, 1, h.shape[-1]))(h)
h = GlobalAveragePooling2D()(h)

projection_1 = Dense(hidden_layer_sizes[0])(h)
projection_1 = Activation("relu")(projection_1)
projection_2 = Dense(hidden_layer_sizes[1])(projection_1)
projection_2 = Activation("relu")(projection_2)
projection_3 = Dense(hidden_layer_sizes[2])(projection_2)

resnet_simclr = Model(inputs, projection_3)
resnet_simclr.summary()

def D1_dot_similarity(x, z, temperature):
  return tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(z, 2)) / temperature

def D2_dot_similarity(x, z, temperature):
  return tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(z), 0), axes = 2) / temperature

def train_step(xi, xj, model, optimizer, batch_size, criterion, negative_mask, temperature):
  with tf.GradientTape() as tape:
    zi = model(xi)
    zj = model(xj)

    zi = tf.math.l2_normalize(zi, axis = 1)
    zj = tf.math.l2_normalize(zj, axis = 1)

    l_pos = D1_dot_similarity(zi, zj, temperature)
    l_pos = tf.reshape(l_pos, (batch_size, 1))

    negatives = tf.concat([zj, zi], axis = 0)

    loss = 0

    for positives in [zi, zj]:
      l_neg = D2_dot_similarity(positives, negatives, temperature)

      labels = tf.zeros(batch_size, dtype = tf.int32)

      l_neg = tf.boolean_mask(l_neg, negative_mask)
      l_neg = tf.reshape(l_neg, (batch_size, -1))

      logits = tf.concat([l_pos, l_neg], axis = 1)
      loss += criterion(y_pred = logits, y_true = labels)

    loss = loss / (2 * batch_size)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss

def train_epoch(model, references, max_queue_size, optimizer, criterion, augmentation_info, negative_mask, use_tqdm, temperature=0.1):

  # Shuffle the references
  np.random.shuffle(references)

  # Instantiate an empty queue
  queue = []

  # Create the storage mechanism, filled with all of the references
  storage = [i for i in np.copy(references)]

  # Get the size of the dataset
  num_references = len(references)

  # Get the strength of the augmentors
  strength = augmentation_info['strength']

  # Get the probabilities of each augmentation
  probs = augmentation_info['probabilities']
  p_jitter = probs['jitter']
  p_blur = probs['gaussian blur']
  p_grayscale = probs['color drop']
  p_flip = probs['flip']

  # Create the augmentors
  queue_preparation_augmentor = augmentors.get_queue_entry_augmentor(strength, p_jitter, p_blur)
  queue_augmentor = augmentors.get_queue_augmentor(strength, p_jitter, p_blur)
  training_augmentor = augmentors.get_training_augmentor(strength, p_jitter, p_blur, p_flip, p_grayscale)

  # Creating a list to store losses
  global epoch_wise_losses, loss
  epoch_wise_losses = []

  def augment_a(current_batch):
    global a
    if use_tqdm:
      a = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(training_augmentor.augment(i)) for i in tqdm(current_batch, "Augmenting 1/2")]).batch(batch_size)
    else:
      a = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(training_augmentor.augment(i)) for i in current_batch]).batch(batch_size)
  def augment_b(current_batch):
    global b
    if use_tqdm:
      b = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(training_augmentor.augment(i)) for i in tqdm(current_batch, "Augmenting 2/2")]).batch(batch_size)
    else:
      b = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(training_augmentor.augment(i)) for i in current_batch]).batch(batch_size)

  def prepare_next_batch(queue, make_new):
    global a, b, current_batch

    if make_new:
      prepare_new(queue)

    # Popping the current batch from the front of the queue
    print ("Popping the current batch from the front of the queue")
    current_batch = []
    for i in range(batch_size):
      current_batch.append(queue.pop(0))

    # Augmenting the current batch
    threads = [
               threading.Thread(target = augment_a, args = (current_batch,)),
               threading.Thread(target = augment_b, args = (current_batch,))
    ]

    for thread in threads:
      thread.start()

    for thread in threads:
      thread.join()

  def train_current_batch(a, b, step_wise_losses):
    # Training on the current batch
    if use_tqdm:
      for xi, xj in zip(tqdm(a, "Training", total = 1), b):
        loss = train_step(xi, xj, model, optimizer, batch_size, criterion, negative_mask, temperature).numpy()
        print ("Loss:", loss)
        step_wise_losses.append(loss)
    else:
      for xi, xj in zip(a, b):
        loss = train_step(xi, xj, model, optimizer, batch_size, criterion, negative_mask, temperature).numpy()
        print ("Loss:", loss)
        step_wise_losses.append(loss)

  def train_current_and_prepare_next(a, b, queue, step_wise_losses, make_new):
    a_train = a
    b_train = b

    threads = [
                threading.Thread(target = prepare_next_batch, args = (queue, make_new,)),
                threading.Thread(target = train_current_batch, args = (a_train, b_train, step_wise_losses,))
    ]

    for thread in threads:
      thread.start()

    for thread in threads:
      thread.join()

  def prepare_new(queue):
    global r, x
    # Popping the first queue_fill_rate * batch_size references from the storage mechanism
    print ("Popping new references from the storage mechanism")
    r = []
    for i in range(queue_fill_rate * batch_size):
      r.append(storage.pop(0))

    # Mapping the references to their corresponding images, augmenting them, and pushing them to the back of the queue
    print ("Downloading images corresonding to new references")
    global x
    x = batch_downloader.download(r).numpy()
    print ("Augmenting new images and pushing them to the queue")
    for i in x:
      queue.append(queue_preparation_augmentor.augment(i))

  
  # While there are sufficient images remaining to train for another step
  while len(storage) >= max_queue_size:
    # Creating a list to store the losses of the current step
    step_wise_losses = []
    # Pop new references from the storage mechanism, map them to their corresponding images, augment them, and push them to the queue
    # Prepare initial batch beforehand
    prepare_next_batch(queue, make_new = True)
    # While there is enough space remaining in the queue to push another set of queue_fill_rate batches
    while len(queue) + queue_fill_rate * batch_size <= max_queue_size:

      # Train on the current batch while simultaneously preparing the next one
      train_current_and_prepare_next(a, b, queue, step_wise_losses, make_new = True)

      # Augmenting the queue
      if use_tqdm:
        queue = [queue_augmentor.augment(i) for i in tqdm(queue, "Augmenting the queue")]
      else:
        queue = [queue_augmentor.augment(i) for i in queue]

    while len(queue) > 0:
      # Train on the current batch while simultaneously preparing the next one
      train_current_and_prepare_next(a, b, queue, step_wise_losses, make_new = False)

      # Augmenting the queue
      if use_tqdm:
        queue = [queue_augmentor.augment(i) for i in tqdm(queue, "Augmenting the queue")]
      else:
        queue = [queue_augmentor.augment(i) for i in queue]

    epoch_wise_losses.append(np.mean(step_wise_losses))
    model.save("simclr_storage_queue/resnet_simclr.h5")
    clear_output()
    print ("{}% Completed.".format((num_references - len(storage)) / num_references * 100))
  
  return model, epoch_wise_losses

def get_negative_mask():
  negative_mask = np.ones((batch_size, 2 * batch_size), dtype = bool)

  for i in range(batch_size):
    negative_mask[i, i] = 0
    negative_mask[i, i + batch_size] = 0

  return tf.constant(negative_mask)

"""
initial_learning_rate = 0.1
decay_steps = 100
lr_decayed_fn = CosineDecay(initial_learning_rate = initial_learning_rate, decay_steps = decay_steps)
optimizer = SGD(lr_decayed_fn)
"""

temperature = 0.1
epochs = 1000

criterion = SparseCategoricalCrossentropy(from_logits = True, reduction = Reduction.SUM)
optimizer = SGD(learning_rate = 0.1)

negative_mask = get_negative_mask()

initial_to_final_augmentation_info = {
    'strength': (0.35, 0.75),
    'probabilities': {
        'jitter': (0.2, 0.8),
        'gaussian blur': (0.125, 0.5),
        'color drop': (0.05, 0.2),
        'flip': (0.125, 0.5)
    }
}

augmentation_info = {
    'strength': initial_to_final_augmentation_info['strength'][0],
    'probabilities': {key: value for key, value in zip(initial_to_final_augmentation_info['probabilities'].keys(), [augmentation[0] for augmentation in initial_to_final_augmentation_info['probabilities'].values()])}
}


"""
num_parallel_trainers = 2

threadLimiter = threading.BoundedSemaphore(num_parallel_trainers)
num_images_per_trainer = (((max_queue_size // batch_size) - 1) * queue_fill_rate) * batch_size
trainer_references = references[:len(references) - (len(references) % num_images_per_trainer)]
trainer_references = [trainer_references[i:i + num_images_per_trainer] for i in range(0, len(trainer_references), num_images_per_trainer)]

threads = []
for i in trainer_references:
  thread = TrainerThread()
  threads.append(thread)

for thread in threads:
  thread.run(resnet_simclr, storage, max_queue_size, optimizer, criterion, augmentation_info, negative_mask, True, temperature)
"""

resnet_simclr, epoch_wise_losses = train_epoch(resnet_simclr, references, max_queue_size, optimizer, criterion, augmentation_info, negative_mask, False, temperature)
