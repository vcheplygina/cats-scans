import numpy as np
from PIL import Image
from skimage import io, color
from skimage.filters import gabor
import cv2
import os
from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import urllib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

#Create a random subset of ImageNet of 10 classes, the amount of images per class can be determined by changing the number 100 on lines 58 en 59 to the desired number.

a = open("synset_ids.txt", "r")
b = a.readlines()
a.close()

class_nums = random.sample(range(0, 21841), 10)
id_list=[]
for x in class_nums:
    text = b[x]
    newtext = text[0:9]
    id_list.append(newtext)

# create dictionary containing class label and corresponding synset id (http://image-net.org/explore_popular.php?page=1)
# for sti10 textural pattern are chosen
synset_ids = {'set 1': '{}'.format(id_list[0]),
              'set 2': '{}'.format(id_list[1]),
              'set 3': '{}'.format(id_list[2]),
              'set 4': '{}'.format(id_list[3]),
              'set 5': '{}'.format(id_list[4]),
              'set 6': '{}'.format(id_list[5]),
              'set 7': '{}'.format(id_list[6]),
              'set 8': '{}'.format(id_list[7]),
              'set 9': '{}'.format(id_list[8]),
              'set 10': '{}'.format(id_list[9]),}
              


# code: https://medium.com/coinmonks/how-to-get-images-from-imagenet-with-python-in-google-colaboratory-aeef5c1c45e5
def collect_urls(synset_id):
    """
    :param synset_id: id of synset in WordNets ImageNet
    :return: urls of images in synset class
    """
    page = requests.get(f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={synset_id}')
    # puts the content of the website into the soup variable, each url on a different line
    soup = BeautifulSoup(page.content, 'html.parser')

    str_soup = str(soup)  # convert soup to string so it can be split
    split_urls = str_soup.split('\r\n')  # split so each url is a different possition on a list
    print(len(split_urls))  # print the length of the list so you know how many urls you have

    if len(split_urls)>100:
        number = 100
    else:
        number = len(split_urls)


    few_urls=[]
    img_nums= random.sample(range(0, len(split_urls)), number)



    for x in img_nums:
        few_urls.append(split_urls[x])



    return few_urls


# for the 10 classes defined in synset_ids, collect all urls of the images belonging to the classes
all_classes = []
for synset_id in synset_ids.values():
    urls = collect_urls(synset_id)
    all_classes.append(urls)

def url_to_image(img_url):
    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    # different exceptions added to avoid errors
    """
    :param img_url: url of image that needs to be converted to numpy array
    :return: numpy array of image
    """
    try:
        resp = urllib.request.urlopen(img_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # return the image
        return image
    # create exceptions in case of non-existing URLs or something related
    except urllib.error.HTTPError:
        print('does not exist any more')
        pass
    except urllib.error.URLError:
        print('URL error')
        pass
    except ValueError:
        print('valueError')
        pass
    except ConnectionResetError:
        print('connection err')
        pass
    except TimeoutError:
        print('timeout err')
        pass

# convert all urls of images collected to numpy arrays and store in lists
img_arrays = []
class_nr = 1

for class_list in all_classes:
    class_arrays = []
    url_nr = 1
    for urls in class_list:
        print(url_nr)
        class_arrays.append(url_to_image(urls))
        url_nr += 1
    img_arrays.append(class_arrays)
    class_nr += 1

# remove None values from lists in img_arrays
improved_img_arrays = []
for img_list in img_arrays:
    no_none = [url for url in img_list if url is not None]
    improved_img_arrays.append(no_none)


all_imgs = [item for sublist in improved_img_arrays for item in sublist]  # flatten list of lists of images

# create list with labels corresponding to the number of images per class
all_labels_lists = []
for img_list, label in zip(improved_img_arrays, synset_ids.keys()):
    num_img = len(img_list)
    all_labels_lists.append(num_img * [str(label)])
all_labels = [item for sublist in all_labels_lists for item in sublist]  # flatten list of lists of labels

#save numpy arrays in home directory
np.save(r"D:\BEP\test\all_imgs", all_imgs)
np.save(r"D:\BEP\test\all_labels", all_labels)