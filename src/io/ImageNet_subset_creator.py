from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import urllib

# create dictionary containing class label and corresponding synset id (http://image-net.org/explore_popular.php?page=1)
# for sti10 textural pattern are chosen
synset_ids = {'stone wall': 'n04326547',
              'brick': 'n02897820',
              'honeycomb': 'n03530642',
              'egg': 'n07840804',
              'rock': 'n09416076',
              'fabric': 'n03309808',
              'cloud': 'n09247410',
              'chain': 'n02999410',
              'ocean': 'n09376198',
              'flower': 'n11669921'}


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

    return split_urls


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


# save numpy arrays in home directory
home = ''
np.save(f'{home}/sti10/all_imgs', all_imgs)
np.save(f'{home}/sti10/all_labels', all_labels)
