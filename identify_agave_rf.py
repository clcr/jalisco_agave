import numpy as np
import sklearn.ensemble as ens
import gdal
from osgeo import ogr, osr
import os
import gdal
from osgeo import ogr, osr
import numpy as np
from tempfile import TemporaryDirectory
import sklearn.ensemble as ens
from sklearn.model_selection import cross_val_score
import scipy.sparse as sp
import pickle

import pdb


def create_matching_dataset(in_dataset, out_path,
                            format="GTiff", bands=1, datatype = None):
    """Creates an empty gdal dataset with the same dimensions, projection and geotransform. Defaults to 1 band.
    Datatype is set from the first layer of in_dataset if unspecified"""
    driver = gdal.GetDriverByName(format)
    if datatype is None:
        datatype = in_dataset.GetRasterBand(1).DataType
    out_dataset = driver.Create(out_path,
                                xsize=in_dataset.RasterXSize,
                                ysize=in_dataset.RasterYSize,
                                bands=bands,
                                eType=datatype)
    out_dataset.SetGeoTransform(in_dataset.GetGeoTransform())
    out_dataset.SetProjection(in_dataset.GetProjection())
    return out_dataset


def reshape_raster_for_ml(image_array):
    """Reshapes an array from gdal order [band, y, x] to scikit order [x*y, band]"""
    bands, y, x = image_array.shape
    image_array = np.transpose(image_array, (1, 2, 0))
    image_array = np.reshape(image_array, (x * y, bands))
    return image_array


def get_training_data(image_path, shape_path, attribute="CODE", shape_projection_id=4326):
    """Given an image and a shapefile with categories, return x and y suitable
    for feeding into random_forest.fit.
    Note: THIS WILL FAIL IF YOU HAVE ANY CLASSES NUMBERED '0'
    WRITE A TEST FOR THIS TOO; if this goes wrong, it'll go wrong quietly and in a way that'll cause the most issues
     further on down the line."""
    with TemporaryDirectory() as td:
        shape_projection = osr.SpatialReference()
        shape_projection.ImportFromEPSG(shape_projection_id)
        image = gdal.Open(image_path)
        image_gt = image.GetGeoTransform()
        x_res, y_res = image_gt[1], image_gt[5]
        ras_path = os.path.join(td, "poly_ras")
        ras_params = gdal.RasterizeOptions(
            noData=0,
            attribute=attribute,
            xRes=x_res,
            yRes=y_res,
            outputType=gdal.GDT_Int16,
            outputSRS=shape_projection
        )
        # This produces a rasterised geotiff that's right, but not perfectly aligned to pixels.
        # This can probably be fixed.
        gdal.Rasterize(ras_path, shape_path, options=ras_params)
        rasterised_shapefile = gdal.Open(ras_path)
        shape_array = rasterised_shapefile.GetVirtualMemArray()
        local_x, local_y = get_local_top_left(image, rasterised_shapefile)
        shape_sparse = sp.coo_matrix(shape_array)
        y, x, features = sp.find(shape_sparse)
        training_data = np.empty((len(features), image.RasterCount))
        image_array = image.GetVirtualMemArray()
        image_view = image_array[:,
                    local_y: local_y + rasterised_shapefile.RasterYSize,
                    local_x: local_x + rasterised_shapefile.RasterXSize
                    ]
        for index in range(len(features)):
            training_data[index, :] = image_view[:, y[index], x[index]]
    return training_data, features


def get_local_top_left(raster1, raster2):
    """Gets the top-left corner of raster1 in the array of raster 2; WRITE A TEST FOR THIS"""
    inner_gt = raster2.GetGeoTransform()
    return point_to_pixel_coordinates(raster1, [inner_gt[0], inner_gt[3]])

def point_to_pixel_coordinates(raster, point, oob_fail=False):
    """Returns a tuple (x_pixel, y_pixel) in a georaster raster corresponding to the point.
    Point can be an ogr point object, a wkt string or an x, y tuple or list. Assumes north-up non rotated.
    Will floor() decimal output"""
    # Equation is rearrangement of section on affinine geotransform in http://www.gdal.org/gdal_datamodel.html
    if isinstance(point, str):
        point = ogr.CreateGeometryFromWkt(point)
        x_geo = point.GetX()
        y_geo = point.GetY()
    if isinstance(point, list) or isinstance(point, tuple):  # There is a more pythonic way to do this
        x_geo = point[0]
        y_geo = point[1]
    if isinstance(point, ogr.Geometry):
        x_geo = point.GetX()
        y_geo = point.GetY()
    gt = raster.GetGeoTransform()
    x_pixel = int(np.floor((x_geo - floor_to_resolution(gt[0], gt[1]))/gt[1]))
    y_pixel = int(np.floor((y_geo - floor_to_resolution(gt[3], gt[5]*-1))/gt[5]))  # y resolution is -ve
    return x_pixel, y_pixel


def floor_to_resolution(input, resolution):
    """Returns input rounded DOWN to the nearest multiple of resolution."""
    return input - (input%resolution)


def reshape_ml_out_to_raster(classes, width, height):
    """Reshapes an output [x*y] to gdal order [y, x]"""
    # TODO: Test this.
    image_array = np.reshape(classes, (height, width))
    return image_array


def classify_image(in_image_path, model, out_image_path):
    print("Classifying image")
    image = gdal.Open(in_image_path)
    image_array = image.GetVirtualMemArray()
    features_to_classify = reshape_raster_for_ml(image_array)
    image_array = None
    width = image.RasterXSize
    height = image.RasterYSize
    image = None
    out_chunks = []
    for i, chunk in enumerate(np.array_split(features_to_classify, 10)):
        print("Classifying {0}".format(i))
        chunk_copy = np.copy(chunk)
        out_chunks.append(model.predict(chunk_copy))
    out_classes = np.concatenate(out_chunks)

    image = gdal.Open(in_image_path)
    out_image = create_matching_dataset(image, out_image_path)
    image = None

    out_image_array = out_image.GetVirtualMemArray(eAccess=gdal.GA_Update)
    out_image_array[...] = reshape_ml_out_to_raster(out_classes, width, height)
    out_image_array = None
    out_image = None


def save_model(model, filepath):
    with open(filepath, "wb") as fp:
        pickle.dump(model, fp)


def load_model(filepath):
    with open(filepath, "rb") as fp:
        return pickle.load(fp)


if __name__ == "__main__":

    image_to_classify = "/home/rnanclares/Documents/gis/Agave/agavero_27012019_planet.tif"
    training_image = "/home/rnanclares/Documents/gis/Agave/agavero_28022017_planet.tif"
    training_shape = "/home/rnanclares/Documents/gis/Agave/rough_training_poly.shp"
    out_path = "/home/rnanclares/Documents/gis/Agave/classified_agave.tif"

    model_out_path = "/home/rnanclares/Documents/gis/Agave/model.pkl"

    does_model_exist = False

    if not does_model_exist:
        features, classes = get_training_data(training_image, training_shape, attribute="CLASE")
        model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
            min_samples_split=16, n_estimators=100, n_jobs=-1, class_weight='balanced')
        model.fit(features[0:20000, :], classes[0:20000])
        save_model(model, model_out_path)
    else:
        model = load_model(model_out_path)
    classify_image(image_to_classify, model, out_path)
    
    