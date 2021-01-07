from Utils import differencewriter, sevenplotvisualizer, zeromean, preprocesser, batchgenerator, batchvector, visualizer, computeglcm, zeromean, compareeucdistance, comparecosine, normalize, writevector, normalize3
from Datatoarray import imagenettoarray, chestxraytoarray, pcamtoarray, kimiatoarray, dtdtoarray, stl10toarray, sti10toarray, isic2018toarray
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import io, color
from skimage.filters import gabor, median
from skimage.feature import greycomatrix, greycoprops
import cv2
import os

#Generate batches with indexes, size is set to 128, index numbers are between the second and third argument of 'batchgenerator'

#ISIC2018 batch
x = batchgenerator(128, 34320, 24306)
#STL-10 batch
y = batchgenerator(128, 13000, 0)
#DTD batch
z = batchgenerator(128, 5640, 0)
#STI-10 batch
a = batchgenerator(128, 8060, 0)
#PCam batch
b = batchgenerator(128, 150000, 0)
#Chest x-ray batch
c = batchgenerator(128, 5856, 0)
#ImageNet batch
d = batchgenerator(128, 363, 0)
#kimia batch
e = batchgenerator(128, 960, 0)


#Generate image arrays out of datasets
isicarray = isic2018toarray(x)
stl10array = stl10toarray(y)
sti10array = sti10toarray(a)
pcamarray = pcamtoarray(b)
dtdarray = dtdtoarray(z)
chestxrayarray = chestxraytoarray(c)
imagenetarray = imagenettoarray(d)
kimiaarray = kimiatoarray(e)

#Conversion to greyscale and uint8 if needed
isicgrey = preprocesser(isicarray)
stl10grey = preprocesser(stl10array)
sti10grey = preprocesser(sti10array)
pcamgrey = preprocesser(pcamarray)
dtdgrey = preprocesser(dtdarray)
chestxraygrey = preprocesser(chestxrayarray)
imagenetgrey = preprocesser(imagenetarray)
kimiagrey = preprocesser(kimiaarray)

#Compute grey level co-occurence matrix and derive texture features
#Outputs are lists that contain the feature values of all images in the batch
isiccontrast, isicdissimilarity, isichomogeneity, isicasm, isicenergy, isiccorrelation = computeglcm(original_imgs=isicarray, original_grey=isicgrey, d=[1], a=[0], levels=256)
stl10contrast, stl10dissimilarity, stl10homogeneity, stl10asm, stl10energy, stl10correlation = computeglcm(original_imgs=stl10array, original_grey=stl10grey, d=[1], a=[0], levels=256)
sti10contrast, sti10dissimilarity, sti10homogeneity, sti10asm, sti10energy, sti10correlation = computeglcm(original_imgs=sti10array, original_grey=sti10grey, d=[1], a=[0], levels=256)
pcamcontrast, pcamdissimilarity, pcamhomogeneity, pcamasm, pcamenergy, pcamcorrelation = computeglcm(original_imgs=pcamarray, original_grey=pcamgrey, d=[1], a=[0], levels=256)
dtdcontrast, dtddissimilarity, dtdhomogeneity, dtdasm, dtdenergy, dtdcorrelation = computeglcm(original_imgs=dtdarray, original_grey=dtdgrey, d=[1], a=[0], levels=256)
chestxraycontrast, chestxraydissimilarity, chestxrayhomogeneity, chestxrayasm, chestxrayenergy, chestxraycorrelation = computeglcm(original_imgs=chestxrayarray, original_grey=chestxraygrey, d=[1], a=[0], levels=256)
imagenetcontrast, imagenetdissimilarity, imagenethomogeneity, imagenetasm, imagenetenergy, imagenetcorrelation = computeglcm(original_imgs=imagenetarray, original_grey=imagenetgrey, d=[1], a=[0], levels=256)
kimiacontrast, kimiadissimilarity, kimiahomogeneity, kimiaasm, kimiaenergy, kimiacorrelation = computeglcm(original_imgs=kimiaarray, original_grey=kimiagrey, d=[1], a=[0], levels=256)

#Uncomment to visualize images of an array, code pauses until new window is closed
#visualizer(nrows=3, ncols=4, nums=x, original_imgs=kimiaarray, original_grey_imgs=grey, filtered_imgs=filtered)

#Normalization of all features using zero mean unit variance
nisiccontrast, nstl10contrast, nsti10contrast, npcamcontrast, ndtdcontrast, nchestxraycontrast, nimagenetcontrast, nkimiacontrast = zeromean(isiccontrast, stl10contrast, sti10contrast, pcamcontrast, dtdcontrast, chestxraycontrast, imagenetcontrast, kimiacontrast)
nisicdissimilarity, nstl10dissimilarity, nsti10dissimilarity, npcamdissimilarity, ndtddissimilarity, nchestxraydissimilarity, nimagenetdissimilarity, nkimiadissimilarity = zeromean(isicdissimilarity, stl10dissimilarity, sti10dissimilarity, pcamdissimilarity, dtddissimilarity, chestxraydissimilarity, imagenetdissimilarity, kimiadissimilarity)
nisichomogeneity, nstl10homogeneity, nsti10homogeneity, npcamhomogeneity, ndtdhomogeneity, nchestxrayhomogeneity, nimagenethomogeneity, nkimiahomogeneity = zeromean(isichomogeneity, stl10homogeneity, sti10homogeneity, pcamhomogeneity, dtdhomogeneity, chestxrayhomogeneity, imagenethomogeneity, kimiahomogeneity)
nisicasm, nstl10asm, nsti10asm, npcamasm, ndtdasm, nchestxrayasm, nimagenetasm, nkimiaasm = zeromean(isicasm, stl10asm, sti10asm, pcamasm, dtdasm, chestxrayasm, imagenetasm, kimiaasm)
nisicenergy, nstl10energy, nsti10energy, npcamenergy, ndtdenergy, nchestxrayenergy, nimagenetenergy, nkimiaenergy = zeromean(isicenergy, stl10energy, sti10energy, pcamenergy, dtdenergy, chestxrayenergy, imagenetenergy, kimiaenergy) 
nisiccorrelation, nstl10correlation, nsti10correlation, npcamcorrelation, ndtdcorrelation, nchestxraycorrelation, nimagenetcorrelation, nkimiacorrelation = zeromean(isiccorrelation, stl10correlation, sti10correlation, pcamcorrelation, dtdcorrelation, chestxraycorrelation, imagenetcorrelation, kimiacorrelation)

#Creation of .txt file with all vector statistics 
#Also creation of 'batchvectors' that contain the average value of each feature of each dataset
isicbatchvector = writevector(name = 'isic2018', nums=x, contrast=nisiccontrast, dissimilarity=nisicdissimilarity, homogeneity=nisichomogeneity, asm=nisicasm, energy=nisicenergy, correlation=nisiccorrelation)
stl10batchvector = writevector(name = 'stl10', nums=y, contrast=nstl10contrast, dissimilarity=nstl10dissimilarity, homogeneity=nstl10homogeneity, asm=nstl10asm, energy=nstl10energy, correlation=nstl10correlation)
sti10batchvector = writevector(name = 'sti10', nums=z, contrast=nsti10contrast, dissimilarity=nsti10dissimilarity, homogeneity=nsti10homogeneity, asm=nsti10asm, energy=nsti10energy, correlation=nsti10correlation)
pcambatchvector = writevector(name = 'pcam', nums=z, contrast=npcamcontrast, dissimilarity=npcamdissimilarity, homogeneity=npcamhomogeneity, asm=npcamasm, energy=npcamenergy, correlation=npcamcorrelation)
dtdbatchvector = writevector(name = 'dtd', nums=z, contrast=ndtdcontrast, dissimilarity=ndtddissimilarity, homogeneity=ndtdhomogeneity, asm=ndtdasm, energy=ndtdenergy, correlation=ndtdcorrelation)
chestxraybatchvector = writevector(name = 'chestxray', nums=z, contrast=nchestxraycontrast, dissimilarity=nchestxraydissimilarity, homogeneity=nchestxrayhomogeneity, asm=nchestxrayasm, energy=nchestxrayenergy, correlation=nchestxraycorrelation)
imagenetbatchvector = writevector(name = 'imagenet', nums=z, contrast=nimagenetcontrast, dissimilarity=nimagenetdissimilarity, homogeneity=nimagenethomogeneity, asm=nimagenetasm, energy=nimagenetenergy, correlation=nimagenetcorrelation)
kimiabatchvector = writevector(name = 'kimia', nums=z, contrast=nkimiacontrast, dissimilarity=nkimiadissimilarity, homogeneity=nkimiahomogeneity, asm=nkimiaasm, energy=nkimiaenergy, correlation=nkimiacorrelation)

#Compute euclidian distance between datasets
#Here, only distances for isic are done
eucdistisic_stl10 = compareeucdistance(isicbatchvector, stl10batchvector)
eucdistisic_sti10 = compareeucdistance(isicbatchvector, sti10batchvector)
eucdistisic_pcam = compareeucdistance(isicbatchvector, pcambatchvector)
eucdistisic_dtd = compareeucdistance(isicbatchvector, dtdbatchvector)
eucdistisic_chestxray = compareeucdistance(isicbatchvector, chestxraybatchvector)
eucdistisic_imagenet = compareeucdistance(isicbatchvector, imagenetbatchvector)
eucdistisic_kimia = compareeucdistance(isicbatchvector, imagenetbatchvector)

#Create lists that contain mean values for each feature for each dataset
#Also one list that contains all euclidian distances to one target, here isic
allcontrast = [stl10batchvector[0], dtdbatchvector[0], sti10batchvector[0], chestxraybatchvector[0], pcambatchvector[0], imagenetbatchvector[0]]
alldissimilarity = [stl10batchvector[1], dtdbatchvector[1], sti10batchvector[1], chestxraybatchvector[1], pcambatchvector[1], imagenetbatchvector[1]]
allhomogeneity = [stl10batchvector[2], dtdbatchvector[2], sti10batchvector[2], chestxraybatchvector[2], pcambatchvector[2], imagenetbatchvector[2]]
allasm = [stl10batchvector[3], dtdbatchvector[3], sti10batchvector[3], chestxraybatchvector[3], pcambatchvector[3], imagenetbatchvector[3]]
allenergy = [stl10batchvector[4], dtdbatchvector[4], sti10batchvector[4], chestxraybatchvector[4], pcambatchvector[4], imagenetbatchvector[4]]
allcorrelation = [stl10batchvector[5], dtdbatchvector[5], sti10batchvector[5], chestxraybatchvector[5], pcambatchvector[5], imagenetbatchvector[5]]
allbatchvector = [eucdistisic_stl10, eucdistisic_dtd, eucdistisic_sti10, eucdistisic_chestxray, eucdistisic_pcam, eucdistisic_imagenet]

#Creation of a .txt file that contains a lot of values, used to copy and paste data into Origin or Excel
differencewriter(isicbatchvector, stl10batchvector, dtdbatchvector, sti10batchvector, chestxraybatchvector, pcambatchvector, imagenetbatchvector, kimiabatchvector)

#Create a figure that contains plots of all datasets on x-axis, and feature values on y-axis
sevenplotvisualizer(allcontrast, alldissimilarity, allhomogeneity, allasm, allenergy, allcorrelation, allbatchvector, source='isic')