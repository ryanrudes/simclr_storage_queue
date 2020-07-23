import random
import tensorflow as tf
import threading
from tqdm.notebook import tqdm
import urllib

def download_and_preprocess(url):
  image = tf.image.decode_jpeg(urllib.request.urlopen(url).read(), channels = 3)
  image = tf.image.resize(image, (256, 256))
  image /= 255.0
  return image

class DownloaderThread(threading.Thread):
  def __init__(self, urls):
    super(DownloaderThread, self).__init__()
    self.urls = urls

  def run(self):
    global thread_limiter
    thread_limiter.acquire()
    
    try:
      self.download(self.urls)
    finally:
      thread_limiter.release()

  def download(self, urls):
    global current_batch
    
    for url in urls:
      current_batch.append(download_and_preprocess(url))

class BatchDownloader():
  def __init__(self, batch_size, images_per_thread, max_threads):
    self.batch_size = batch_size
    self.images_per_thread = images_per_thread
    self.max_threads = max_threads
    global thread_limiter
    thread_limiter = threading.BoundedSemaphore(max_threads)

  def download(self, urls):
    global current_batch
    current_batch = []

    threads = []

    for i in tqdm(range(0, len(urls), self.images_per_thread), "Instantiating {} threads.".format(len(urls) // self.images_per_thread)):
      thread = DownloaderThread(urls[i:i + self.images_per_thread])
      threads.append(thread)

    for thread in tqdm(threads, "Starting {} threads.".format(len(threads))):
      thread.start()

    for thread in tqdm(threads, "Joining {} threads.".format(len(threads))):
      thread.join()

    current_batch = tf.convert_to_tensor(current_batch)
    return current_batch
