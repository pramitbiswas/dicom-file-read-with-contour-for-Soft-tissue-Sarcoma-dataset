# This code is used to read dicom dataset given in this link:
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=21266533
#
# The lib available at https://github.com/KeremTurgutlu/dicom-contour will not
# able to read this. Thus, some modification has been made for the particular dataset
# as given below.

from collections import defaultdict
import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from scipy.sparse import csc_matrix


# ROIContourSeq denotes index of roi_names
ROIContourSeq = 0   # Change this as per requirement
# Consider these
# directory Structure:
# main_dir\
# |-dcm\
# |  |-1.000000-RTstructT1-64320\
# |  |-1.000000-RTstructT1-75059\
# |-dicom.py
case_path = "H:\\main_dir\\dcm\\1.000000-RTstructT1-64320\\"

# get the (main) contour file for this patient/scan
# get .dcm files
fpaths = [case_path + f for f in os.listdir(case_path) if f[-4:]=='.dcm']
n = 0; contour_file = None; other_dcm_files = []
for fpath in fpaths:
    f = dicom.read_file(fpath)
    if 'ROIContourSequence' in dir(f):
        # the (main) contour file
        contour_file = fpath.split('\\')[-1]
        n += 1
    else:
        # other .dcm files
        other_dcm_files.append({'name':fpath.split('\\')[-1], 'SOPInstanceUID':f.SOPInstanceUID})
if n > 1: warnings.warn("There are multiple contour files, considering the last one!")
if contour_file is None:
    print("No contour file found in directory")

# read contour data (rt_sequence)
contour_data = dicom.read_file(case_path + contour_file)

# extract roi index in RT Struct
roi_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]

# index=0 (ROIContourSeq) means that we are getting RTV information
RTV = contour_data.ROIContourSequence[ROIContourSeq]

# get contour datasets in a list
contours = [contour for contour in RTV.ContourSequence]

#coord2pixels
img_contour_arrays_infos = []
for cdata in contours:
    contour_coord = cdata.ContourData

    # x, y, z coordinates of the contour in mm
    x0 = contour_coord[len(contour_coord)-3]
    y0 = contour_coord[len(contour_coord)-2]
    z0 = contour_coord[len(contour_coord)-1]
    coord = []
    for i in range(0, len(contour_coord), 3):
        x = contour_coord[i]
        y = contour_coord[i+1]
        z = contour_coord[i+2]
        l = math.sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0))
        l = math.ceil(l*2)+1
        for j in range(1, l+1):
            coord.append([(x-x0)*j/l+x0, (y-y0)*j/l+y0, (z-z0)*j/l+z0])
        x0 = x
        y0 = y
        z0 = z

    # extract the image id corresponding to given countour
    # read that dicom file (assumes filename = sopinstanceuid.dcm)
    img_ID = cdata.ContourImageSequence[0].ReferencedSOPInstanceUID

    file_ID = None
    for item in other_dcm_files:
        if item['SOPInstanceUID']==img_ID:
            file_ID = item['name']
            break
    if file_ID is None:
        print(f'.dcm file is not found for ReferencedSOPInstanceUID: {img_ID}')
    img = dicom.read_file(case_path + file_ID)

    # main image
    img_arr = img.pixel_array

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.round((y - origin_y) / y_spacing),
                    np.round((x - origin_x) / x_spacing)) for x, y, _ in coord]

    # get contour data for the image
    rows = []
    cols = []
    for i, j in list(set(pixel_coords)):
        rows.append(i)
        cols.append(j)
    contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)),
                                dtype=np.int8,
                                shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

    c2p={
            'img_array':img_arr,
            'contour_array':contour_arr,
            'img_ID':img_ID,
            'file_ID':file_ID,
            'ImagePositionPatient':img.ImagePositionPatient
    }
    img_contour_arrays_infos.append(c2p)

# ORDERING THE IMAGES
ordered_img_contour_arrays_infos = sorted(img_contour_arrays_infos, key=lambda k: k['img_ID'])

# get all image-contour array pairs
# multiple contours are not considered
# sum contour arrays and generate new img_contour_arrays
contour_dict = defaultdict(int)
for info in ordered_img_contour_arrays_infos:
    contour_dict[info['img_ID']] += info['contour_array']
image_dict = {}
for info in ordered_img_contour_arrays_infos:
    image_dict[info['img_ID']] = info['img_array']
img_contour_arrays = [(image_dict[k], contour_dict[k], k) for k in image_dict]

# get first image - contour array
first_image, first_contour, img_id = img_contour_arrays[0]

# # show an example
# plt.figure(figsize=(20, 10))
# plt.subplot(1,2,1)
# plt.imshow(first_image)
# plt.subplot(1,2,2)
# plt.imshow(first_contour)
# plt.show()

images = np.array([item['img_array'] for item in ordered_img_contour_arrays_infos])
contours = np.array([item['contour_array'] for item in ordered_img_contour_arrays_infos])

print(images.shape)
print(contours.shape)

def plot_all_2d_contour_gif(images, contours, figsize=(20, 20)):
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=figsize)

    def update(i):
        masked_contour_array = np.ma.masked_where(contours[i] == 0, contours[i])

        plt.subplot(1, 2, 1)
        plt.imshow(images[i].squeeze(), cmap='gray', interpolation='none')
        plt.subplot(1, 2, 2)
        plt.imshow(images[i].squeeze(), cmap='gray', interpolation='none')
        plt.imshow(masked_contour_array, cmap='cool', interpolation='none', alpha=0.7)

    ani = FuncAnimation(fig, update, frames=range(images.shape[0]), interval=250)

    plt.show()

# plot_all_2d_contour_gif(images, contours)
