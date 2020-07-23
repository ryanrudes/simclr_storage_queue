def encode_url(url):
  return "".join(url.split("/")[-3::2]).replace(".JPEG", "")

def decode_url(url):
  folder = url[:url.index("n")]
  synset = url[url.index("n"):url.index("_")]
  number = url[url.index("n"):]
  return "https://storage.googleapis.com/simclr_dataset_bucket/simclr/data/Imagenet/Imagenet%20Divided/" + folder + "/" + synset + "/" + number + ".JPEG"

with open("image_urls.txt", "w") as f:
  f.write("\n".join([decode_url(url) for url in open("~/image_links.txt", "r").read().split("\n")]))

f.close()
