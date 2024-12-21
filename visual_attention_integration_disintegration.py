import os
import cv2
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def plot_image_2_single_focal(path, gaze, chest_bbox):
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x, y = gaze['x_position'], gaze['y_position']

    xmin, ymin, xmax, ymax = chest_bbox['xmin'].item(), chest_bbox['ymin'].item(), chest_bbox['xmax'].item(), chest_bbox['ymax'].item()

    x_final = []
    y_final = []
    for x_point, y_point in zip(x, y):
        if x_point >= xmin and x_point <= xmax and y_point >= ymin and y_point <= ymax:
            x_final.append(x_point)
            y_final.append(y_point)

    x_final.append(0)
    x_final.append(image_array.shape[2])
    y_final.append(0)
    y_final.append(image_array.shape[1])

    points = []
    for x, y in zip(x_final, y_final):
        points.append([x, y])

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    label = kmeans.fit_predict(np.array(points))

    cluster_0 = []
    cluster_1 = []
    idx = 0
    for i in label:
        if i == 0:
            cluster_0.append(points[idx])
        elif i == 1:
            cluster_1.append(points[idx])
        idx += 1

    x_final = []
    y_final = []
    for cluster0_temp in cluster_0:
        x_temp, y_temp = cluster0_temp
        x_final.append(x_temp)
        y_final.append(y_temp)

    x_final.append(0)
    x_final.append(image_array.shape[2])
    y_final.append(0)
    y_final.append(image_array.shape[1])

    plt.imshow(image_array[0, :, :], cmap='gray')
    plt.scatter(x_final, y_final, c='red', s=0.4)

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/focal_image_{}.png'.format(name), dpi=1000)
    plt.clf()

def plot_image_2_single_global(path, gaze, chest_bbox):
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x, y = gaze['x_position'], gaze['y_position']

    xmin, ymin, xmax, ymax = chest_bbox['xmin'].item(), chest_bbox['ymin'].item(), chest_bbox['xmax'].item(), chest_bbox['ymax'].item()

    x_final = []
    y_final = []
    for x_point, y_point in zip(x, y):
        if x_point >= xmin and x_point <= xmax and y_point >= ymin and y_point <= ymax:
            x_final.append(x_point)
            y_final.append(y_point)

    x_final.append(0)
    x_final.append(image_array.shape[2])
    y_final.append(0)
    y_final.append(image_array.shape[1])

    plt.imshow(image_array[0, :, :], cmap='gray')
    plt.scatter(x_final, y_final, c='red', s=0.4)

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/global_image_{}.png'.format(name), dpi=1000)
    plt.clf()

def plot_visual_attention_heatmap_2_single_global(path, gaze, chest_bbox):
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x, y = gaze['x_position'], gaze['y_position']

    xmin, ymin, xmax, ymax = chest_bbox['xmin'].item(), chest_bbox['ymin'].item(), chest_bbox['xmax'].item(), chest_bbox['ymax'].item()

    x_final = []
    y_final = []
    for x_point, y_point in zip(x, y):
        if x_point >= xmin and x_point <= xmax and y_point >= ymin and y_point <= ymax:
            x_final.append(x_point)
            y_final.append(y_point)

    x_final.append(0)
    x_final.append(image_array.shape[2])
    y_final.append(0)
    y_final.append(image_array.shape[1])

    heatmap, xedges, yedges = np.histogram2d(x_final, y_final, bins=(image_array.shape[1], image_array.shape[2]))
    heatmap = gaussian_filter(heatmap, sigma=64)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = cv2.rotate(heatmap.T, cv2.ROTATE_90_CLOCKWISE)
    heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE)

    plt.imshow(image_array[0, :, :], cmap='gray')
    # plt.imshow(heatmap, extent=extent, origin='lower', cmap=cm.jet)
    plt.scatter(x_final, y_final, c='red', s=0.4)

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/global_{}.png'.format(name), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_visual_attention_heatmap_2_single_focal(path, gaze, chest_bbox):
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x, y = gaze['x_position'], gaze['y_position']

    xmin, ymin, xmax, ymax = chest_bbox['xmin'].item(), chest_bbox['ymin'].item(), chest_bbox['xmax'].item(), chest_bbox['ymax'].item()

    x_final = []
    y_final = []
    for x_point, y_point in zip(x, y):
        if x_point >= xmin and x_point <= xmax and y_point >= ymin and y_point <= ymax:
            x_final.append(x_point)
            y_final.append(y_point)

    x_final.append(0)
    x_final.append(image_array.shape[2])
    y_final.append(0)
    y_final.append(image_array.shape[1])

    points = []
    for x, y in zip(x_final, y_final):
        points.append([x, y])

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    label = kmeans.fit_predict(np.array(points))

    cluster_0 = []
    cluster_1 = []
    idx = 0
    for i in label:
        if i == 0:
            cluster_0.append(points[idx])
        elif i == 1:
            cluster_1.append(points[idx])
        idx += 1

    x_final = []
    y_final = []
    for cluster0_temp in cluster_0:
        x_temp, y_temp = cluster0_temp
        x_final.append(x_temp)
        y_final.append(y_temp)

    x_final.append(0)
    x_final.append(image_array.shape[2])
    y_final.append(0)
    y_final.append(image_array.shape[1])

    heatmap, xedges, yedges = np.histogram2d(x_final, y_final, bins=(image_array.shape[1], image_array.shape[2]))
    heatmap = gaussian_filter(heatmap, sigma=128)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = cv2.rotate(heatmap.T, cv2.ROTATE_90_CLOCKWISE)
    heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE)

    plt.imshow(image_array[0, :, :], cmap='gray')
    # plt.imshow(heatmap, extent=extent, origin='lower', cmap=cm.jet)
    plt.scatter(x_final, y_final, c='red', s=0.4)

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/focal_{}.png'.format(name), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_visual_attention_heatmap_1_integration_approach1(path, x, y):
    image_array = cv2.imread(path)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x_final = x
    y_final = y

    x_final.append(0)
    x_final.append(image_array.shape[1])
    y_final.append(0)
    y_final.append(image_array.shape[0])

    heatmap, xedges, yedges = np.histogram2d(x_final, y_final, bins=(image_array.shape[0], image_array.shape[1]))
    heatmap = gaussian_filter(heatmap, sigma=128)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = cv2.rotate(heatmap.T, cv2.ROTATE_90_CLOCKWISE)
    heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE)

    # plt.imshow(image_array, cmap='gray')
    # plt.imshow(lung_segment)
    plt.imshow(heatmap, extent=extent, origin='lower', cmap=cm.jet)
    # plt.scatter(x_final, y_final, c='red', s=0.4)

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/integration/{}.png'.format(name), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_visual_attention_heatmap_1_disintegration_approach1(path, x, y):
    image_array = cv2.imread(path)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x_final = x
    y_final = y

    x_final.append(0)
    x_final.append(image_array.shape[1])
    y_final.append(0)
    y_final.append(image_array.shape[0])

    heatmap, xedges, yedges = np.histogram2d(x_final, y_final, bins=(image_array.shape[0], image_array.shape[1]))
    heatmap = gaussian_filter(heatmap, sigma=64)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = cv2.rotate(heatmap.T, cv2.ROTATE_90_CLOCKWISE)
    heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE)

    # plt.imshow(image_array[0, :, :], cmap='gray')
    # plt.imshow(lung_segment)
    plt.imshow(heatmap, extent=extent, origin='lower', cmap=cm.jet)
    # plt.scatter(x_final, y_final, c='red', s=0.4)

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/disintegration/{}.png'.format(name), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_visual_attention_heatmap_1_integration_approach2(path, x, y):
    image_array = cv2.imread(path)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x_final = x
    y_final = y

    x_final.append(0)
    x_final.append(image_array.shape[1])
    y_final.append(0)
    y_final.append(image_array.shape[0])

    heatmap, xedges, yedges = np.histogram2d(x_final, y_final, bins=(image_array.shape[0], image_array.shape[1]))
    heatmap = gaussian_filter(heatmap, sigma=128)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = cv2.rotate(heatmap.T, cv2.ROTATE_90_CLOCKWISE)
    heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE)

    # plt.imshow(image_array, cmap='gray')
    # plt.imshow(lung_segment)
    plt.imshow(heatmap, extent=extent, origin='lower', cmap=cm.jet)
    # plt.scatter(x_final, y_final, c='red', s=0.4)

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/integration/{}.png'.format(name), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_visual_attention_heatmap_1_disintegration_approach2(path, x, y):
    image_array = cv2.imread(path)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x_final = x
    y_final = y

    x_final.append(0)
    x_final.append(image_array.shape[1])
    y_final.append(0)
    y_final.append(image_array.shape[0])

    heatmap, xedges, yedges = np.histogram2d(x_final, y_final, bins=(image_array.shape[0], image_array.shape[1]))
    heatmap = gaussian_filter(heatmap, sigma=64)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = cv2.rotate(heatmap.T, cv2.ROTATE_90_CLOCKWISE)
    heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE)

    # plt.imshow(image_array[0, :, :], cmap='gray')
    # plt.imshow(lung_segment)
    plt.imshow(heatmap, extent=extent, origin='lower', cmap=cm.jet)
    # plt.scatter(x_final, y_final, c='red', s=0.4)

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/disintegration/{}.png'.format(name), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_visual_attention_heatmap_1_focal_timewindowed(path, x, y, tw=0):
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x_final = x
    y_final = y

    x_final = np.append(x_final, 0)
    x_final = np.append(x_final, image_array.shape[1])
    y_final = np.append(y_final, 0)
    y_final = np.append(y_final, image_array.shape[0])

    heatmap, xedges, yedges = np.histogram2d(x_final, y_final, bins=(image_array.shape[1], image_array.shape[2]))
    heatmap = gaussian_filter(heatmap, sigma=128)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = cv2.rotate(heatmap.T, cv2.ROTATE_90_CLOCKWISE)
    heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE)

    super_imposed_img = cv2.addWeighted(heatmap, 0.5, image_array[:, :, 0], 0.5, 0)
    plt.imshow(image_array[:, :, 0])

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/heatmaps_0_globalfocal_timewindowed_{}/focal_{}.png'.format(tw, name), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_visual_attention_heatmap_1_global_timewindowed(path, x, y, tw=0):
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x_final = x
    y_final = y

    x_final = np.append(x_final, 0)
    x_final = np.append(x_final, image_array.shape[1])
    y_final = np.append(y_final, 0)
    y_final = np.append(y_final, image_array.shape[0])

    heatmap, xedges, yedges = np.histogram2d(x_final, y_final, bins=(image_array.shape[1], image_array.shape[2]))
    heatmap = gaussian_filter(heatmap, sigma=64)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = cv2.rotate(heatmap.T, cv2.ROTATE_90_CLOCKWISE)
    heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE)

    super_imposed_img = cv2.addWeighted(heatmap, 0.5, image_array[:, :, 0], 0.5, 0)
    plt.imshow(heatmap)

    # plt.imshow(image_array[:, :, 0], cmap='gray')
    # plt.imshow(heatmap, extent=extent, origin='lower', cmap=cm.jet)

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/heatmaps_0_globalfocal_timewindowed_{}/global_{}.png'.format(tw, name), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_visual_attention_heatmap_2_focal_timewindowed(path, x, y, tw=0):
    image_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x_final = x
    y_final = y

    x_final = np.append(x_final, 0)
    x_final = np.append(x_final, image_array.shape[1])
    y_final = np.append(y_final, 0)
    y_final = np.append(y_final, image_array.shape[0])

    heatmap, xedges, yedges = np.histogram2d(x_final, y_final, bins=(image_array.shape[0], image_array.shape[1]))
    heatmap = gaussian_filter(heatmap, sigma=128)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    print(heatmap.shape, image_array.shape)

    image_array = image_array.astype(float)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_array, cmap='gray', aspect='auto')  # Display X-ray in grayscale
    ax.imshow(heatmap, alpha=0.5, aspect='auto')  # Heatmap with transparency

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/heatmaps_1_globalfocal_timewindowed_{}/focal_{}.png'.format(tw, name), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_visual_attention_heatmap_2_global_timewindowed(path, x, y, tw=0):
    image_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    name = path.split('/')[-1]
    name = name.split('.')[0]

    x_final = x
    y_final = y

    x_final = np.append(x_final, 0)
    x_final = np.append(x_final, image_array.shape[1])
    y_final = np.append(y_final, 0)
    y_final = np.append(y_final, image_array.shape[0])

    heatmap, xedges, yedges = np.histogram2d(x_final, y_final, bins=(image_array.shape[0], image_array.shape[1]))
    heatmap = gaussian_filter(heatmap, sigma=64)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    image_array = image_array.astype(float)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_array, cmap='gray', aspect='auto')  # Display X-ray in grayscale
    ax.imshow(heatmap, alpha=0.5, aspect='auto')  # Heatmap with transparency

    plt.axis('off')
    plt.show()
    plt.savefig('/path/to/heatmaps_1_globalfocal_timewindowed_{}/global_{}.png'.format(tw, name), dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.clf()

def preprocess_1_globalfocal():
    master_path = pd.read_csv('/path/to/eye-gaze-data-for-chest-x-rays-1.0.0/master_sheet.csv')
    gaze = pd.read_csv('/path/to/eye-gaze-data-for-chest-x-rays-1.0.0/eye_gaze.csv')

    gaze = gaze[['DICOM_ID', 'X_ORIGINAL', 'Y_ORIGINAL']]
    gaze = gaze.groupby('DICOM_ID').agg(list)

    gaze['DICOM_ID'] = gaze.index

    count = 0
    for id, x, y in tqdm(zip(gaze['DICOM_ID'], gaze['X_ORIGINAL'], gaze['Y_ORIGINAL'])):
        try:
            image_path = '/home/moibhattacha/datasets/eye_gaze_cxr_1_png/'+id+'.png'
            plot_visual_attention_heatmap_1_global(image_path, x, y)
            plot_visual_attention_heatmap_1_focal(image_path, x, y)
        except:
            flag = 0

def preprocess_1_globalfocal_timewindowed():
    master_path = pd.read_csv('/path/to/eye-gaze-data-for-chest-x-rays-1.0.0/master_sheet.csv')
    gaze = pd.read_csv('/path/to/eye-gaze-data-for-chest-x-rays-1.0.0/eye_gaze.csv')

    gaze = gaze[['DICOM_ID', 'X_ORIGINAL', 'Y_ORIGINAL', 'Time (in secs)']]
    gaze = gaze.groupby('DICOM_ID').agg(list)

    gaze['DICOM_ID'] = gaze.index

    for id, x, y, time in tqdm(zip(gaze['DICOM_ID'], gaze['X_ORIGINAL'], gaze['Y_ORIGINAL'], gaze['Time (in secs)'])):
        try:
            image_path = '/path/to/eye_gaze_cxr_1/'+id+'.dcm'
            range_0 = (0, time[-1]/4.)
            range_1 = (time[-1]/4., time[-1]/2.)
            range_2 = (time[-1]/2., time[-1]*(3/4))
            range_3 = (time[-1]*(3/4), time[-1])

            time_0 = np.vectorize(lambda time: range_0[0] <= time <= range_0[1])(time)
            time_1 = np.vectorize(lambda time: range_1[0] <= time <= range_1[1])(time)
            time_2 = np.vectorize(lambda time: range_2[0] <= time <= range_2[1])(time)
            time_3 = np.vectorize(lambda time: range_3[0] <= time <= range_3[1])(time)

            x_0, y_0 = np.array(x)[np.array(time_0)], np.array(y)[np.array(time_0)]
            x_1, y_1 = np.array(x)[np.array(time_1)], np.array(y)[np.array(time_1)]
            x_2, y_2 = np.array(x)[np.array(time_2)], np.array(y)[np.array(time_2)]
            x_3, y_3 = np.array(x)[np.array(time_3)], np.array(y)[np.array(time_3)]

            plot_visual_attention_heatmap_1_global_timewindowed(image_path, x_0, y_0, tw=0)
            plot_visual_attention_heatmap_1_global_timewindowed(image_path, x_1, y_1, tw=1)
            plot_visual_attention_heatmap_1_global_timewindowed(image_path, x_2, y_2, tw=2)
            plot_visual_attention_heatmap_1_global_timewindowed(image_path, x_3, y_3, tw=3)

            plot_visual_attention_heatmap_1_focal_timewindowed(image_path, x_0, y_0, tw=0)
            plot_visual_attention_heatmap_1_focal_timewindowed(image_path, x_1, y_1, tw=1)
            plot_visual_attention_heatmap_1_focal_timewindowed(image_path, x_2, y_2, tw=2)
            plot_visual_attention_heatmap_1_focal_timewindowed(image_path, x_3, y_3, tw=3)
        except:
            flag = 0

def preprocess_2_single_globalfocal():
    master_path_3 = pd.read_csv('/path/to/reflacx-reports-and-eye-tracking-data-for-localization-of-abnormalities-in-chest-x-rays-1.0.0/main_data/metadata_phase_3.csv')

    for id, image in tqdm(zip(master_path_3['id'], master_path_3['image'])):
        try:
            # if image.split('/')[-1] == '005aff0f-0c236062-06df954a-25ad1874-bcdffcb0.dcm':
                gaze = pd.read_csv('/path/to/reflacx-reports-and-eye-tracking-data-for-localization-of-abnormalities-in-chest-x-rays-1.0.0/main_data/'+id+'/fixations.csv')
                image_path = '/path/to/eye_gaze_cxr_2/'+image.split('/')[-1]
                chest_bbox = pd.read_csv('/path/to/reflacx-reports-and-eye-tracking-data-for-localization-of-abnormalities-in-chest-x-rays-1.0.0/main_data/'+id+'/chest_bounding_box.csv')
                plot_visual_attention_heatmap_2_single_global(image_path, gaze, chest_bbox)
                plot_visual_attention_heatmap_2_single_focal(image_path, gaze, chest_bbox)
                # plot_image_2_single_global(image_path, gaze, chest_bbox)
                # plot_image_2_single_focal(image_path, gaze, chest_bbox)
        except:
            flag = 0

def preprocess_2_single_globalfocal_timewindowed():
    master_path_3 = pd.read_csv('/path/to/reflacx-reports-and-eye-tracking-data-for-localization-of-abnormalities-in-chest-x-rays-1.0.0/main_data/metadata_phase_3.csv')

    for id, image in tqdm(zip(master_path_3['id'], master_path_3['image'])):
            if image.split('/')[-1] == 'f7216f70-ea0d379f-f2d9028e-817459ef-f8357a3e.dcm':
                gaze = pd.read_csv('/path/to/reflacx-reports-and-eye-tracking-data-for-localization-of-abnormalities-in-chest-x-rays-1.0.0/main_data/'+id+'/fixations.csv')
                image = image.split('/')[-1]
                print(image.split('.dcm')[0]+'.png')
                image_path = '/path/to/eye_gaze_cxr_2_png/'+image.split('.dcm')[0]+'.png'
                chest_bbox = pd.read_csv('/path/to/reflacx-reports-and-eye-tracking-data-for-localization-of-abnormalities-in-chest-x-rays-1.0.0/main_data/'+id+'/chest_bounding_box.csv')

                x, y = gaze['x_position'], gaze['y_position']

                time = gaze['timestamp_start_fixation'].values.tolist()
                range_0 = (0, time[-1]/4.)
                range_1 = (time[-1]/4., time[-1]/2.)
                range_2 = (time[-1]/2., time[-1]*(3/4))
                range_3 = (time[-1]*(3/4), time[-1])

                time_0 = np.vectorize(lambda time: range_0[0] <= time <= range_0[1])(time)
                time_1 = np.vectorize(lambda time: range_1[0] <= time <= range_1[1])(time)
                time_2 = np.vectorize(lambda time: range_2[0] <= time <= range_2[1])(time)
                time_3 = np.vectorize(lambda time: range_3[0] <= time <= range_3[1])(time)

                x_0, y_0 = np.array(x)[np.array(time_0)], np.array(y)[np.array(time_0)]
                x_1, y_1 = np.array(x)[np.array(time_1)], np.array(y)[np.array(time_1)]
                x_2, y_2 = np.array(x)[np.array(time_2)], np.array(y)[np.array(time_2)]
                x_3, y_3 = np.array(x)[np.array(time_3)], np.array(y)[np.array(time_3)]

                plot_visual_attention_heatmap_2_global_timewindowed(image_path, x_0, y_0, tw=10)
                plot_visual_attention_heatmap_2_global_timewindowed(image_path, x_1, y_1, tw=11)
                plot_visual_attention_heatmap_2_global_timewindowed(image_path, x_2, y_2, tw=12)
                plot_visual_attention_heatmap_2_global_timewindowed(image_path, x_3, y_3, tw=13)

                plot_visual_attention_heatmap_2_focal_timewindowed(image_path, x_0, y_0, tw=10)
                plot_visual_attention_heatmap_2_focal_timewindowed(image_path, x_1, y_1, tw=11)
                plot_visual_attention_heatmap_2_focal_timewindowed(image_path, x_2, y_2, tw=12)
                plot_visual_attention_heatmap_2_focal_timewindowed(image_path, x_3, y_3, tw=13)

                break

if __name__ == "__main__":
    flag = 0
