# %%

import os
import rasterio.plot
from rasterio.plot import show
import tensorflow as tf
import random
import cv2 as cv 
import rasterio
from rasterio.transform import from_origin
from rasterio import features
import pandas as pd
import math 
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon
from mining_pages_utils.tensorflow_utils import run_vesselprofile_segmentation, readNoextensionImage, build_detectfn
from tensorflow import keras
import glob
from typing import List
from abc import ABCMeta, abstractmethod
import requests
from typing import Tuple
from mining_pages_utils.request_utils import addModifiedEntry
import spatial_efd

import datetime


from mining_pages_utils.request_utils import getListOfDBs
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from pathlib import Path
from descartes import PolygonPatch
from typing import Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
from PIL import Image, ImageColor
from IPython.display import display
import datetime
import numpy as np
from shutil import copyfile
import requests
import rasterio.mask
import json

# %%
def mask_to_polygons_layer(mask):
    all_polygons = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask >0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        #return shapely.geometry.shape(shape)
        all_polygons.append(shapely.geometry.shape(shape))
        #print('What does the mask_10:layer give us?', shape, value)
        


    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons

# %%
def polygons_to_mask(geodf, image_np):
    if 'geometry' in geodf.columns:
    
        #print('Polygonsto Mask Geometry is valid!')
        out_image = features.rasterize(geodf['geometry'], out_shape=(image_np.shape[0],image_np.shape[1]), default_value= 255)
        return out_image

# %%
def close_holes(series, smallerthanarea):
    """
    Close polygon holes by limitation to the exterior ring.
    Args:
        poly: Input shapely Polygon
    Example:
        df.geometry.apply(lambda p: close_holes(p))
    
    """
    exterior=Polygon(series['geometry'].exterior.coords)
    if series['geometry'].geom_type == 'Polygon':
        if series['geometry'].interiors:
            largeinteriors = []
            for interior in series['geometry'].interiors:
                interior=Polygon(interior.coords)
                print('This is the interior areasize: ', interior.area)
                print('This is the smallerthanarea: ', smallerthanarea)
                if interior.area > smallerthanarea:
                    print('This interior will make a difference')
                    exterior = exterior.difference(interior)
    series['geometry']=exterior
    return series
        
   
 



# %%
def getarea(series):
    series['area']=series['geometry'].area
    print('This is the Area: ', series['area'])
    return series

# %%
def bufferpoly (series, buffer):
    series['geometry']=series['geometry'].buffer(buffer)
    return series

# %%
def multi2singlepolygon (multipolygons):
    listpolygons = []
    if multipolygons.geom_type == 'MultiPolygon':

        # extract polygons out of multipolygon
        for polygon in multipolygons.geoms:
            if polygon.type == 'Polygon':
                exterior_coords = polygon.exterior.coords[:]
                interior_coords = []
                for interior in polygon.interiors:
                    interior_coords += interior.coords[:]
            if polygon.geom_type == 'Polygon':

                #print('But now I am not anymore or?', type(polygon))
                listpolygons.append(polygon)

    if multipolygons.geom_type == 'GeometryCollection':

        # extract polygons out of multipolygon
        for polygon in multipolygons.geoms:

            if polygon.geom_type == 'Polygon':

                listpolygons.append(polygon)
    elif multipolygons.geom_type == 'Polygon':
        listpolygons.append(multipolygons)
    else:
        print('What kind of Geomerty are you? Answer: ',multipolygons.geom_type)
    


    return listpolygons

# %%
def multi2singlepolygon_df (gdf):
    newgdf = gpd.GeoDataFrame()
    for index, row in gdf.iterrows():

        if row['geometry'].geom_type == 'MultiPolygon':
            print('I am Multipolygon')
            # extract polygons out of multipolygon
            for polygon in row['geometry'].geoms:
                if polygon.type == 'Polygon':
                    exterior_coords = polygon.exterior.coords[:]
                    interior_coords = []
                    for interior in polygon.interiors:
                        interior_coords += interior.coords[:]
                    print('This is the interiors of multi2single:', interior_coords)
                if polygon.geom_type == 'Polygon':
                    print(vars(polygon))
                    #print('But now I am not anymore or?', type(polygon))
                    singlepolyrow = row
                    singlepolyrow['geometry']=polygon
                    newgdf = newgdf.append(singlepolyrow)

        if row['geometry'].geom_type == 'GeometryCollection':
            print('I am GeometryCollection')
            # extract polygons out of multipolygon
            for polygon in row['geometry'].geoms:
                print('I am an elemnt of GeomCollection: ', polygon.geom_type)
                if polygon.geom_type == 'Polygon':
                    print('But now I am not anymore or?', type(polygon))
                    singlepolyrow = row
                    singlepolyrow['geometry']=polygon
                    newgdf = newgdf.append(singlepolyrow)
        elif row['geometry'].geom_type == 'Polygon':
            singlepolyrow = row
            singlepolyrow['geometry']=row['geometry']
            newgdf = newgdf.append(singlepolyrow)
        else:
            print('What kind of Geomerty are you? Answer: ',row['geometry'].geom_type)
    


    return newgdf

# %%
def simplify(series, d, cf):
    print('Simplify tolerance distance and cf: ', d, cf)
    series['geometry'] = series['geometry'].buffer(-d).buffer(d*cf).intersection(series['geometry']).simplify(d)
   
    return series
    

# %%
def filterintersects(tobefilteredgdf, referencegdf, largerthanthis):
    filteredgdf = gpd.GeoDataFrame()
    for index, targetrow in tobefilteredgdf.iterrows():
        for index, referencerow in referencegdf.iterrows():
            if targetrow['geometry'].intersects(referencerow['geometry']) or targetrow['geometry'].area > largerthanthis :
                filteredgdf = filteredgdf.append(targetrow)
    return filteredgdf
    

# %%
def process_files(vessel):
    print(vessel)
    image_np = readNoextensionImage(vessel)
    print(type(image_np))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = miningfiguresdetectfn(input_tensor)
    image_np_with_detections = image_np.copy()
    empty_with_detections = np.zeros((image_np.shape[0], image_np.shape[1], image_np.shape[2])).astype(np.uint8)
    
    #num_detections = int(output_dict.pop('num_detections'))
    #print(num_detections)
    # The following processing is only for single image
    detection_boxes = tf.squeeze(detections['detection_boxes'], [0])
    detection_masks = tf.squeeze(detections['detection_masks'], [0])
    #print(detection_boxes)
    
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(detections['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0],
                            [real_num_detection, -1])
    #print('detection_boxes', detection_boxes)
    detection_masks = tf.slice(detection_masks, [0, 0, 0],
                            [real_num_detection, -1, -1])
    #print('detection_masks', detection_masks)
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    detections['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    detections['num_detections'] = int(detections['num_detections'][0])
    detections['detection_classes'] = detections['detection_classes'][0].numpy(
    ).astype(np.uint8)
    detections['detection_boxes'] = detections['detection_boxes'][0].numpy()
    detections['detection_scores'] = detections['detection_scores'][0].numpy()
    detections['detection_masks'] = detections['detection_masks'][0].numpy()

    #print(detections['detection_scores'])
    #print('image-size: ',image_np.shape)
    #print('mask-size: ',detections['detection_masks'].shape)


 



    vis_util.visualize_boxes_and_labels_on_image_array(
        empty_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        instance_masks=detections['detection_masks'],
        use_normalized_coordinates=True,
        max_boxes_to_draw=3,
        min_score_thresh=.1,
        mask_alpha=1,
        skip_boxes=True,
        skip_scores= True,
        skip_labels=True)


    
    mask = empty_with_detections[:,:,:1]
    if cv.countNonZero(mask) == 0 :
        print('CountNonZero:', cv.countNonZero(mask))
        plt.imshow(image_np)
        #plt.show()
        
    if not cv.countNonZero(mask) == 0 :
        detectedmask = mask.astype(np.uint8)
        mask = mask.astype(np.uint8)
        
        kernelwidth = int(image_np.shape[0] * image_np.shape[1] / 4000)
        kernelheight = int(image_np.shape[0] * image_np.shape[1] / 2000)
        kernel = np.ones((kernelheight,kernelwidth),np.uint8)
        
        print('firstkernel',kernel.shape)
        mask = cv.dilate(mask,kernel,iterations = 1)
        kernelwidth = int(math.sqrt(image_np.shape[0] * image_np.shape[1]) / 50)
        kernelheight = int(math.sqrt(image_np.shape[0] * image_np.shape[1]) / 50)
        kernel = np.ones((kernelwidth,kernelheight),np.uint8)
        print('This is the Kernel to erode the detectedmask: ', kernel.shape)
        detectedmask = cv.erode(detectedmask,kernel,iterations = 1)


        mask[mask > 0] = cv.GC_PR_FGD
        mask[mask == 0] = cv.GC_BGD
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        (mask, bgModel, fgModel) = cv.grabCut(image_np, mask, None, bgModel,fgModel, 10, mode=cv.GC_INIT_WITH_MASK)
        values = (
            ("Definite Background", cv.GC_BGD),
            ("Probable Background", cv.GC_PR_BGD),
            ("Definite Foreground", cv.GC_FGD),
            ("Probable Foreground", cv.GC_PR_FGD),
        )
        
        valuemask = (mask == cv.GC_PR_FGD).astype("uint8") * 255
        detectedmask = detectedmask.astype("uint8") * 255
        valuemask = cv.addWeighted(detectedmask, 0.5, valuemask, 0.5, 0.0)

        kernelwidth = int(image_np.shape[0] * image_np.shape[1] / 80000)
        kernelheight = int(image_np.shape[0] * image_np.shape[1] / 80000)
        kernel = np.ones((kernelwidth,kernelheight),np.uint8)
        #valuemask = cv.dilate(valuemask,kernel,iterations = 1)
        multipolygons = mask_to_polygons_layer(valuemask)
        #print('This is the length of mask_to_polygons resullt (grabcut): ', multipolygons)
        detectedmask_polys = mask_to_polygons_layer(detectedmask)

        
        print('Here comes grabcut multi2single:')
        listpolygons = multi2singlepolygon(multipolygons)
        print('This is the length of multi2singlepolygon resullt (grabcut): ', len(listpolygons))
        print('Here comes detected multi2single:')
        detected_listpolygons = multi2singlepolygon(detectedmask_polys)

        
        detectedpolygons = [{'geometry':polygon, 'method':'detected'} for polygon in detected_listpolygons]
        detecteddf = pd.DataFrame(detectedpolygons)
        detectedgdf = gpd.GeoDataFrame(detecteddf, geometry='geometry')        

        #listpolygons = list(multipolygons.geoms)
        polygons = [{'geometry':polygon, 'method':'grabcut'} for polygon in listpolygons]
        
        df = pd.DataFrame(polygons)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        gdf = gdf.apply(close_holes, smallerthanarea=int(image_np.shape[0] * image_np.shape[1] / 1000), axis=1)
        #buffer = -image_np.shape[0] * image_np.shape[1] / 160000
        #gdf = gdf.apply(bufferpoly, buffer=buffer, axis=1)     
        gdf=gdf.apply(getarea, axis=1)
        gdfmax= gdf.nlargest(5, 'area', keep='first')

        #print ('tolerance=', str(image_np.shape[0] * image_np.shape[1] / 160000))
        #gdfmax = gdfmax.apply(simplify, d=1.5, cf=1.6, axis=1)
        buffer = -int(math.sqrt(image_np.shape[0] * image_np.shape[1]) / 200)
        print('This is the Buffer ',buffer)
        gdfmax = gdfmax.apply(bufferpoly, buffer=buffer, axis=1)
        buffer = int(math.sqrt(image_np.shape[0] * image_np.shape[1]) / 200)
        gdfmax = gdfmax.apply(bufferpoly, buffer=buffer, axis=1)
        gdfmax = multi2singlepolygon_df(gdfmax)
        gdfmax=gdfmax.apply(getarea, axis=1)
        print('Only larger than this survives:', int(image_np.shape[0] * image_np.shape[1] / 300))
        gdfmax=gdfmax[gdfmax['area'] > int(image_np.shape[0] * image_np.shape[1] / 300)]
        gdfmax = gdfmax.apply(close_holes, smallerthanarea=int(image_np.shape[0] * image_np.shape[1] / 1000), axis=1)
        gdfmax=gdfmax.apply(getarea, axis=1)
        print(gdfmax)
        if not gdfmax.empty and 'geometry' in gdfmax.columns:
            gdfmax = filterintersects(gdfmax, detectedgdf, largerthanthis=max(gdfmax['area'])/30)
            gdfmax=gdfmax.apply(getarea, axis=1)
            gdfmax= gdf.nlargest(1, 'area', keep='first')
       


            #gpd.GeoDataFrame(gdf, geometry='geometry')
            

            transform = from_origin(0, 0, 1, 1)

            image_raster = rasterio.open(vessel, 'r', driver='PNG',transform=transform)
            #gdfmax = gpd.GeoDataFrame(gdfmax, geometry='geometry')

        
            #rasterio.plot.show((image_raster, 1))
            out_image = polygons_to_mask(gdfmax, empty_with_detections)
            show(out_image)
            #plt.show()
            original_img = Image.fromarray(image_np)
            segimage = out_image
            segpolygons = gdfmax['geometry']
            #max(multipolygon, key=lambda a: a.area)
            fig, ax = plt.subplots(figsize=(15,15))
            rasterio.plot.show(image_raster, ax=ax)
            gdfmax.plot(ax=ax, facecolor='none', edgecolor='red')
            detectedgdf.plot(ax=ax, facecolor='none', edgecolor='green')
            #show(ax)
            
            #plt.show()#
            return original_img, segimage, segpolygons



        #print(type(polygons))
        
        #show(out_image)
        #plt.show()

        #kernelwidth = int(image_np.shape[0] / 120)
        #kernelheight = int(image_np.shape[1] / 120)
        #kernel = np.ones((kernelwidth,kernelheight),np.uint8)
        #mask = cv.erode(mask,kernel,iterations = 1)

        #plt.figure()
        #plt.imshow(valuemask)
        #plt.show()
	
    #ret,thresh1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)
    #(mask, bgModel, fgModel) = cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount[, mode])  


    #pil_mask = Image.fromarray(np.uint8(255.0*0.4*(detections['detection_masks'] > 0))).convert('L')
    #pil_mask = pil_image.convert('RGB')
    #np.copyto(image, np.array(pil_image.convert('RGB')))
    #segmented_img = Image.fromarray(pil_mask)

    

        

# %%
def createVOCfolder (outputpath, taskname,labelmappath):
    originals = taskname + '_originals'
    originalimagesfolder = outputpath / originals
    taskfolder = outputpath / taskname
    targetlabelmappath = taskfolder / 'labelmap.txt'
    imagesetfolder = taskfolder / 'ImageSets'
    imagesetsegmentationfolder = imagesetfolder / 'Segmentation'
    subsettextfile = imagesetsegmentationfolder / 'train.txt'
    segmentclassfolder = taskfolder / 'SegmentationClass'
    segmentobjectfolder = taskfolder / 'SegmentationObjekt' 
    print(taskfolder)
    taskfolder.mkdir(parents=True, exist_ok=True)
    originalimagesfolder.mkdir(parents=True, exist_ok=True)
    segmentclassfolder.mkdir(parents=True, exist_ok=True)
    segmentobjectfolder.mkdir(parents=True, exist_ok=True)
    imagesetfolder.mkdir(parents=True, exist_ok=True)
    imagesetsegmentationfolder.mkdir(parents=True, exist_ok=True)
    with open(subsettextfile, 'w') as fp:
        pass
    
    copyfile(labelmappath, targetlabelmappath)
    return originalimagesfolder, taskfolder, subsettextfile, segmentclassfolder, segmentobjectfolder

# %%
def addToVOCfolder (original_img, segmented_img):

    #pil_mask = pil_image.convert('RGB')
    #np.copyto(image, np.array(pil_image.convert('RGB')))
    vesselbasename = os.path.basename(vessel) + '.png'
    print('This is the segmented image:' , segmented_img.shape, type(segmented_img))
    segmented_img = Image.fromarray(segmented_img)
    segmented_img.save(os.path.join(segmentclassfolder, vesselbasename), format='png')
    original_img.save(os.path.join(originalimagesfolder, vesselbasename), format='png')
    
    with subsettextfile.open('a+') as f:
        f.write(os.path.basename(vessel) + "\n")
    
    

# %%
def selectimages(amount ):
    if amount == 'all':
        vessel_image_list = [ os.path.join(data_path, i) for i in os.listdir(data_path)]

    if type(amount) == int:


        vessel_image_list = [ os.path.join(data_path, i) for i in random.sample(os.listdir(data_path), amount)]
    #print(vessel_image_list)
    return vessel_image_list

# %%
class FeatureEntry(List):
    List[float]

# %%
class ResnetFeatureVectors ():
    """
    @brief: Data generator to return feature vector of given keras model
    @param images_path: location of source images
    @param model: path to keras Model
    @param image: size to rescale images to
    """

    def __init__(self, model: keras.Model, segmented_img: np.ndarray, image_size: Tuple[int, int] = (512, 512)) -> None:
        self._model = model
        self._segmented_img = segmented_img
        self._image_size = image_size

    def get(self):
        self.__init__
        img = cv.resize(cv.cvtColor(self._segmented_img, cv.COLOR_GRAY2RGB),self._image_size)
        #cv.CvtColor(vis0, vis2, cv.CV_GRAY2BGR)
        feature_vec = self._model.predict(
            img[np.newaxis, ...], verbose=False)
        print('ResnetFeatureVectors produce with get this:', feature_vec)
        return feature_vec.flatten().tolist()
        

    

# %%
def compute_resnet_featurevector(segmented_img):
    model = keras.applications.resnet50.ResNet50(
        include_top=False, pooling='avg', weights='imagenet')
    vector_generator = ResnetFeatureVectors(model,segmented_img).get()
    return vector_generator


# %%
class FourierDescriptorBase(metaclass=ABCMeta):
    """
    Base class to construct Fourier descriptor of given binary images \n
    @param list of filenames
    @param normalize set if Fourier descriptors should be normalized
    @param descriptor_harmonics number of fourier descriptor (FD) harmonics -> FD[-m_n, -m_n-1, ...-m_1, m_0, m_1, ...., m_n-1, m_n ] for n harmonics
    """

    def __init__(self, segmented_img, segpolygons, descriptor_harmonics: int, normalize: bool = True):
        self._segmented_img = segmented_img
        self._segpolygons = segpolygons
        self.descriptors = []
        self.contours = []
        self.descriptor_harmonics = descriptor_harmonics
        self._normalize = normalize

    def __iter__(self) -> FeatureEntry:
        image = self._segmented_img
        self.contours.append(
            self.detectOutlineContour(image.astype('uint8')))
        segpolygons = self._segpolygons
        

        unnormalized_descriptor = self.makeFourierDescriptorFromPolygon(
           segpolygons= self._segpolygons, harmonics=self.descriptor_harmonics)
        if self._normalize:
            yield FeatureEntry(self.normalizeDescriptor(unnormalized_descriptor).tolist())
        else:
            yield FeatureEntry(unnormalized_descriptor.tolist())


    def ylist(self, contours):
        xlist = []
        for contour in contours:
            xlist = xlist + contour[-1][:, 1]
        return ylist

    def __len__(self) -> int:
        return len(self.images)

    def detectOutlineContour(self, bin_image: np.ndarray) -> np.ndarray:
        """ Detects outline contour of given binary image """
        contours, _ = cv.findContours(
            bin_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        return self.contours

    def makeFourierDescriptor_basic(self, contour: List):
        """ calculates the (unnormalized!) fourier descriptor from a list of points """
        contour_complex = np.empty(contour.shape[:-1], dtype=complex)
        contour_complex.real = contour[:, 0]
        contour_complex.imag = contour[:, 1]
        return np.fft.fft(contour_complex)

    @abstractmethod
    def normalizeDescriptor(self, descriptor: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def getDescriptorDimensions(self) -> int:
        pass

    def getLargestContour(self, contours: List) -> np.ndarray:
        """ Returns the largest contour of a list of contours. 
            Makes sure that contours created of noisy image artifacts are not returned
        """
        contours = sorted(contours, key=len, reverse=True)
        return np.array(contours[0][:, 0, :])

    def makeFourierDescriptorFromPolygon(self, segpolygons, harmonics: int = 40) -> np.ndarray:
        """ @brief  Compute the Fourier Descriptors for a polygon.
                    Implements Kuhl and Giardina method of computing the coefficients
                    for a specified number of harmonics. See the original paper for more detail:
                    Kuhl, FP and Giardina, CR (1982). Elliptic Fourier features of a closed
                    contour. Computer graphics and image processing, 18(3), 236-258.
                    Or see W. Burger et. al. - Principles of Digital Image Processing - Advanced Methods
            @param X (list): A list (or numpy array) of x coordinate values.
            @param Y (list): A list (or numpy array) of y coordinate values.
            @param harmonics (int): The number of harmonics to compute for the given
                    shape, defaults to 10.
            @return numpy.ndarray: A complex numpy array of shape (harmonics, ) representing the unnormalized Fourier descriptor 
        """
        print('This should be harmonics:', harmonics)
        new_vector_length = 2*harmonics+1
        FD = np.zeros(new_vector_length)+1j*np.zeros(new_vector_length)

        contour = np.array([(x, y) for x, y in zip(X, Y)])
        contour=segpolygons

        N = len(contour)
        dxy = np.array([contour[(i+1) % N] - contour[i]for i in range(N)])
        dt = np.sqrt((dxy ** 2).sum(axis=1))
        t = np.concatenate([([0, ]), np.cumsum(dt)])
        T = t[-1]

        # compute coefficient G(0)
        a0 = 0
        c0 = 0
        for i in range(len(contour)):
            s = (t[i+1]**2 - t[i]**2)/(2*dt[i]) - t[i]
            a0 += s*dxy[i, 0] + dt[i]*(X[i]-X[0])
            c0 += s*dxy[i, 1] + dt[i]*(Y[i]-Y[0])
        FD[0] = np.complex(X[0] + T**-1*a0, Y[0] + T**-1*c0)

        # compute remaining coefficients
        for m in range(1, harmonics+1):
            omega0 = (2*np.pi*m)/T * np.array([t[i]
                                               for i in range(len(contour))])  # t
            omega1 = (
                2*np.pi*m)/T * np.array([t[(i+1) % len(contour)] for i in range(len(contour))])

            a_m = np.sum((np.cos(omega1) - np.cos(omega0)) / dt * dxy[:, 0])
            c_m = np.sum((np.cos(omega1) - np.cos(omega0)) / dt * dxy[:, 1])

            b_m = np.sum((np.sin(omega1) - np.sin(omega0)) / dt * dxy[:, 0])
            d_m = np.sum((np.sin(omega1) - np.sin(omega0)) / dt * dxy[:, 1])

            const = T/(2*np.pi*m)**2
            FD[m] = const * np.complex(a_m+d_m, c_m - b_m)
            FD[-m % new_vector_length] = const * \
                np.complex(a_m - d_m, c_m + b_m)

        return FD




# %%
class FourierDescriptorPhase(FourierDescriptorBase):
    """
    Class to construct phase preserving Fourier descriptor of given binary images \n
    @param list of filenames or list of images
    @param number of fourier descriptor (FD) harmonics -> FD[-m_n, -m_n-1, ...-m_1, m_0, m_1, ...., m_n-1, m_n ] for n harmonics
    @return pair of Fourier descriptors G1 and G2
    """

    def normalizeDescriptor(self, descriptor: np.ndarray) -> np.ndarray:
        self._setTranslationInvariant(descriptor)
        self._setScaleInvariant(descriptor)
        G_a, G_b = self._setStartPointInvariant(descriptor)

        # rotation

        G_ac = np.concatenate((G_a.real, G_a.imag))
        G_bc = np.concatenate((G_b.real, G_b.imag))
        complex_vector = np.concatenate((G_ac, G_bc))

        return complex_vector.real + complex_vector.imag

    def _setTranslationInvariant(self, descriptor: np.ndarray):
        """
        @brief Makes given descriptor translation invariant
        @param descriptor Fourier descriptor
        """
        descriptor[0] = 0

    def _setScaleInvariant(self, descriptor: np.ndarray):
        """
        @brief Makes given descriptor scale invariant
        @param descriptor Fourier descriptor
        """
        s = 0
        for m in range(1, self.descriptor_harmonics+1):
            s += np.abs(descriptor[-m]) + np.abs(descriptor[m])
        v = 1.0/np.sqrt(s)

        for m in range(1, self.descriptor_harmonics+1):
            descriptor[-m] *= v
            descriptor[m] *= v

    def _setStartPointInvariant(self, descriptor: np.ndarray) -> np.ndarray:
        """
        @brief Make Fourier Descriptor invariant to start point phase phi and phi + np.pi
        @param descriptor Fourier descriptor
        """
        phi = self._getStartPointPhase(descriptor)
        G_a = self._shiftStartPointPhase(descriptor, phi)
        G_b = self._shiftStartPointPhase(descriptor, phi + np.pi)

        return G_a, G_b

    def _getStartPointPhase(self, descriptor: np.ndarray) -> float:
        """
        @brief  Returns start point phase phi by maximizing function _fp(descriptor,phi), with phi [0,np.pi)
                The maximum is simple brute-force search (OPTIMIZE!!)
        @param descriptor Fourier descriptor
        """
        c_max = -float("inf")
        phi_max = 0
        K = 400  # brute force with 400 steps TO DO: OPTIMIZE!!

        for k in range(K):
            phi = np.pi * float(k)/K
            c = self._fp(descriptor, phi)
            if c > c_max:
                c_max = c
                phi_max = phi

        return phi_max

    def _fp(self, descriptor: np.ndarray, phi: float):
        """
        @brief Look for quantity that depends only on the phase differneces within the Fourier descriptor pairs
        @param descriptor Fourier descriptor
        @param phi start point phase
        """
        s = 0
        for m in range(1, self.descriptor_harmonics+1):
            z1 = descriptor[-m]*np.exp(- 1j*m*phi)
            z2 = descriptor[m]*np.exp(1j*m*phi)
            s += z1.real * z2.imag - z1.imag * z2.real
        return s

    def _shiftStartPointPhase(self, descriptor: np.ndarray, phi: float) -> np.ndarray:
        """
        @brief normalizes discriptor by shifting start point phase
        @param descriptor Fourier descriptor
        @param phi start point phase
        """
        G = np.copy(descriptor)
        for m in range(1, self.descriptor_harmonics+1):
            G[-m] = descriptor[-m] * np.exp(-1j*m*phi)
            G[m] = descriptor[m] * np.exp(1j*m*phi)

        return G

    def getDescriptorDimensions(self) -> int:
        """
        @brief For n harmonics => FD[-m_n, ...., 0, ...., m_n ]
                We have to FDs (G_a, G_b)
                We separate in real part and imaginary part 
        """
        return (2*self.descriptor_harmonics+1)*2**2


# %%
def put_data_in_pouchdb(url, auth, featurevectors):

    doc_url = url
    res = requests.get(doc_url, auth=auth)
    if res.status_code != 404:
        payload = res.json()
        rev = payload['_rev']
        
        if not 'featureVectors' in payload['resource'].keys():
            payload['resource']['featureVectors'] = featurevectors
        if 'featureVectors' in payload['resource'].keys():
            for featurevectorkey in featurevectors.keys():
                payload['resource']['featureVectors'][featurevectorkey] = featurevectors[featurevectorkey]
        payload = addModifiedEntry(payload)
        #json_object = json.loads(payload)
        #json_formatted_str = json.dumps(payload, indent=2)
        #print(json_formatted_str)
        
        stat = requests.put(f"{doc_url}?rev={rev}", auth=auth, json=payload)
        return print(stat)
    else:
        return print(res.status_code)

# %%
def calculateEFD(polygons):
    listofEFDcoeffs = []
    for polygon in polygons:
        print()
        x, y = polygon.exterior.coords.xy
        
        #x, y, centroid = spatial_efd.ProcessGeometryNorm(segpolygons[0])
        harmonic = 80
        coeffs = spatial_efd.CalculateEFD(x, y, harmonic)
        #coeffs, rotation = spatial_efd.normalize_efd(coeffs, size_invariant=True)
        #fig, ax = plt.subplots(figsize=(15,15))
        #spatial_efd.spatial_efd.plotComparison(ax, coeffs, harmonic, x, y, rotation=rotation, color1='k', width1=1, color2='r', width2=1)
        #plt.show()
        #spatial_efd.spatial_efd.inverse_transform(coeffs, locus=(0, 0), n_coords=300, harmonic=20)
        listofEFDcoeffs = listofEFDcoeffs + coeffs.flatten().tolist()
        #fourierfeatures = FourierDescriptorPhase(segmented_img, segpolygons, 40)
        #for fet in fourierfeatures:
    return listofEFDcoeffs

# %%
seg_model = "E:/Traindata/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8_profilesegmentv11/saved_model"
PATH_TO_LABELS = "E:/Traindata/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8_profilesegmentv10/labelmap.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
db_url = 'http://localhost:3000'
auth = ('', 'blub')
alldbs = getListOfDBs(db_url, auth)
selectlist = ['tallzira_ed']
list = [item for item in alldbs if item.endswith('_ed') or item.endswith('_edv2') ]
list = [item for item in list if not item.endswith('ock_ed')]
print(list)
imagestore = 'E:/mining_shapes/imagestore/'
miningfiguresdetectfn = build_detectfn(seg_model)



# %%
selectlist = ['lattara6_edv2', 'sidikhrebish_ed','simitthus_ed','althiburos_ed']
selectlist50 = ['althiburos_ed','lattara6_edv2', 'sidikhrebish_ed','simitthus_ed','bonifay2004_ed', 'hayes1972_edv2', 'sabratha_ed', 'tallzira_ed']
outputpath = Path("E:/mining_shapes/segmented_profiles/")
vessel_image_list = []
taskname = 'ImproveMaskTrain2'
labelmappath =  Path("E:/Traindata/Trainingdata_fromCVAT/profile_segmentation/voc/labelmap.txt")


for db_name in ['idaishapes_test']:
    data_path = imagestore + db_name +'/'
    vessel_image_list_db = selectimages('all')
    #
    vessel_image_list = vessel_image_list + vessel_image_list_db

originalimagesfolder, taskfolder, subsettextfile, segmentclassfolder, segmentobjectfolder = createVOCfolder(outputpath, taskname,labelmappath)
#print(vocfolder)
with tqdm(total=len(vessel_image_list)) as pbar:
    for vessel in vessel_image_list:
        #print(vessel)
        try:
            original_img, segmented_img, segpolygons = process_files(vessel)
        except:
            print('No Mask detected...')
        addToVOCfolder(original_img, segmented_img)
        print("Compute feature vectors")
        drawingID = os.path.basename(vessel)
        db_name = os.path.basename(os.path.dirname(vessel))
        drawingURL = f'{db_url}/{db_name}/{drawingID}'
        print('RessourceURL for idaifield:', drawingURL)
        resnetfeatures = compute_resnet_featurevector(segmented_img)
        FDs = calculateEFD(segpolygons)
        print('Length of phaseFourier:', len(FDs))
        print (type(FDs))
        put_data_in_pouchdb(url = drawingURL, auth=auth, featurevectors = {'resnet':resnetfeatures,'phaseFourier':FDs})
        pbar.update(1)


# %% [markdown]
# 


