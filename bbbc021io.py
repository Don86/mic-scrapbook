import numpy as np
import os
import csv
import re

from skimage import io, filters, util, data, img_as_float

#Dependency
import patch_extraction as pex

eps = np.finfo(float).eps


def get_main_list(path):
    """Extracts a clean list of all the subfolders in given path.
    Omits hidden files"""
    subfolders = os.listdir(path)
    subfolders2 = []

    #Screen the subfolder names for only the relevant ones
    #That is, only those with plate numbers in them
    for subfolder in subfolders:
        if "Week" in subfolder:
            subfolders2.append(subfolder)

    return subfolders2


def get_platenum(fn):
    """Extracts the plate number from the given file name

    Params
    ------
    fn: str. Filename

    Returns
    -------
    platenum: str
    """
    pattern = re.compile(r'_[\d]*')
    num = re.findall(pattern, fn)
    if len(num) > 1:
        print("Warning: multiple regex matches for plate number found")
        print("Default behaviour: use first match")
    platenum = num[0].strip('_')

    return platenum


def get_plate_well(fn):
    """Extracts the plate number from the given file name

    Params
    ------
    fn: str. File name.

    Returns
    -------
    platewell: str. Plate-number/well-number coordinates.

    """
    pattern = re.compile(r'_[\d]*')
    pnum = re.findall(pattern, fn)
    if len(pnum) > 1:
        print("Warning: %s matches for plate number found" % len(pnum))
        print("Default behaviour: use first match")
    platenum = pnum[0].strip('_')

    pattern2 = re.compile(r'_[A-z][0-9][0-9]_')
    wn = re.findall(pattern2, fn)
    if len(wn) > 1:
        print("Warning: %s matches for plate number found" % len(wn))
        print("Default behaviour: use first match")
    wellnum = wn[0].strip('_')

    platewell = platenum + '-' + wellnum

    return platewell


def get_subfolder_patch_data(subpath, n_patches, patch_len, db, verbose=1):
    """At the subfolder level, get the platenum.
    At image level, get well numbers and image data from all files in the given
    subfolder path

    Params
    ------
    subpath: str. Path of subfolder
    n_patches: int. Desired no. of patches to extract from each image.
    patch_len: int. Desired side length of each patch.
    db: pandas database from which to query labels based on plate-well coords:
        compound, concentration and moa.
    verbose: int, 0, 1 or 2
        0: All comments suppressed
        1: Only prints comments at subfolder level
        2: Prints all comments

    Return
    ------
    subfolder_data_arr: array. each row = [patches, plate num, well num]
        plate num will be the same for all rows
    """
    plate = get_platenum(subpath)
    sf_list = os.listdir(subpath)
    if verbose > 0:
        print("%s images found in subfolder" % len(sf_list))

    sf_data_list = []
    idx = 1
    for fn in sf_list:
        patches, wn = get_image_patches_and_wellnum(fn, subpath, n_patches, patch_len)
        plate_well = 'BBBC021-'+plate+'-'+wn
        result = search_labels(plate_well, db)
        if len(result) == 1:
            c1 = result[0][1]; c2 = result[0][2]; moa = result[0][3]
            img_data_list = [patches, plate_well, c1, c2, moa]
            sf_data_list.append(img_data_list)
            if verbose > 1:
                print("%s. %s: %s-%s, moa: %s" % (idx, plate_well, c1, c2, moa))
        elif len(result) == 0:
            img_data_list = []
            if verbose > 1:
                print("Warning: no results found for %s, ommitting plate-well" % plate_well)
        elif len(result) > 1:
            c1 = result[0][1]; c2 = result[0][2]; moa = result[0][3]
            img_data_list = [patches, plate_well, c1, c2, moa]
            sf_data_list.append(img_data_list)
            if verbose > 1:
                print("Warning: %s results found for %s, returning first row." % (len(result), plate_well))
                print("%s. %s: %s-%s, moa: %s" % (idx, plate_well, c1, c2, moa))

        idx +=1

    return sf_data_list


def get_image_patches_and_wellnum(fn, subpath, n_patches, patch_len):
    """Operates at image-level.
    Get the image data itself, and the well num from the filename itself
    Uses some patch extraction method, instead of imread wholesale
    so that not the entire image is extracted

    Returns
    -------
    patches: array. Patches of the image. Shape (n_patches, patch_len, patch_len, ch)
    wellnum: str. Well number as extracted from the filename itself
    """

    #Get the image data
    fn_path = subpath + "/" + fn
    img = img_as_float(io.imread(fn_path)[:,:,:3])
    im_height, im_width, ch = img.shape

    #for the moment, let's just generate new patch coordinates for each image
    #We'll think about using the same patch coords across all images later
    coords = pex.generate_patch_coords(n_patches, patch_len, im_width, im_height)
    patches = pex.extract_patches_3d(img, coords, patch_len)

    #Get well numbers, which typically appear between '_', e.g.'_B07_'
    #Or at the front of the file name, e.g. 'B06_'
    pattern = re.compile(r'_[A-z][0-9][0-9]_|^[A-z][0-9][0-9]_')
    wn = re.findall(pattern, fn)
    if len(wn) > 1:
        print("Warning: multiple matches for well num found")
        print("Default behaviour: use first match")
    wellnum = wn[0].strip('_')

    return patches, wellnum


def search_labels(platewell, db1):
    """Search for the compound-concentration labels in a db,
    given the plate-well coordinates

    Params
    ------
    platewell: str, format 'BBBC021-<plate no.>-<well no.>
    db1: a pandas dataframe,
        with columns 'plate-well', 'compound', 'concentration', 'moa'

    Returns
    -------
    results: array of search results. Each row = 1 result:
        (plate_well, compound, concentration, moa)
    """

    query = [platewell]
    result = db1.loc[db1['plate-well'].isin(query)]
    result = np.array(result)

    return result


if __name__ == '__main__':
    import patch_extraction as pex
    print('accuracy: %.3f' % np.mean((yts_pred == yts).astype(float)))
