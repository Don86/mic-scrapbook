from skimage import io, filters, data
import microscopium as mic
import microscopium.features as micfeat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import os
import time

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import SparseCoder
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

"""Various methods that are frequently used.
"""

def note_to_self():
    print("Clean this up; partition between utility functions for supervised, unsupervised, or agnostic methods")


def get_feat_values(gene_values, pathstr):
	"""
	Given a list of names of gene_values, output feature vector values
	for all genes in the list.
	Takes awhile to run.
	
	Parameters
	----------
	gene_values : list of string
        The filenames of the images
	pathstr : string
		Directory to search in for filenames
		
	Return
	------
	f_vec : list
		a list of lists of feature values of each filename
	"""
	
	f_vec = []
	
	for i in range(len(gene_values)):
		image_url = pathstr + '//' + gene_values[i]
		image = io.imread(image_url)
		al, bl = micfeat.default_feature_map(image)
		f_vec.append(al)
	
	f_arr = np.array(f_vec)
	
	return f_arr

	
def get_feat_distribution(feat_list, index):
	"""get all the values of a single feature
	Defunct?
	"""
	
	d0 = np.array([])
	for i in range(len(feat_list)):
		d0.np.append(feat_list[i][index])
	
	return d0

	
def get_normalized_features(X):
	"""Normalize the columns of X to a standard normal distribution
	
	Parameters
	----------
	X : 2D array of float, shape (M, N)
		Input data, M rows (features) and N columns (samples)
		
	Returns
	-------
	Xn : 2D array of float, shape (M, N)
		Normalized data
	"""
	
	X_mean = np.average(X,axis = 0)
	X_std = np.std(X,axis = 0)
	
	#Initialize Xn
	(M,N) = X.shape
	Xn = np.zeros((M,N))
	
	#Fill Xn
	for r in range(len(X)):
		for c in range(len(X[0])):
			if X_std[r] != 0:
				Xn[r][c] = (X[r][c] - X_mean[r])/X_std[r]
			else: #if s.d. = 0, i.e. a constant r.v.
				Xn[r][c] = X[r][c]
	
	return Xn
	
	
def quantile_norm(X):
	"""Source: Juan's book, chap. 1
	Normalize the columns of X to each have the same distribution. 
	
	Given an expression matrix (microarray data, read counts, etc.) of M genes
	by N samples, quantile normalization ensures all samples have the same spread
	of data.
	
	The input data is first log-transformed, then the data across each row is 
	averaged to obtain an average column. Each column quantile is replaced with 
	the corresponding quantile of the average column.
	The data is then transformed back to counts.
	
	Parameters
	----------
	X : 2D array of float, shape (M, N)
		Input data, M rows (genes/features) and N columns (samples)
		
	Returns
	-------
	Xn : 2D array of float, shape (M, N)
		Normalized data
	"""
	
	#log-transform the data:
	logX = np.log2(X + 1)
	
	#compute quantiles
	log_quantiles = np.mean(np.sort(logX, axis=0), axis=1)
	# compute the column-wise ranks. Each observation is replaced with its
	# rank in that column: the smallest observation is replaced by 0, the
	# second-smallest by 1, ..., and the largest by M, the number of rows.
	
	ranks = np.transpose([np.round(stats.rankdata(col)).astype(int) - 1
	for col in X.T])
	
	# index the quantiles for each rank with the ranks matrix
	logXn = log_quantiles[ranks]
	
	# convert the data back to counts (casting to int is optional)
	Xn = np.round(2**logXn - 1).astype(int)

	return Xn
	
def intragene_pairwise_distance(X):
	"""For an input array of shape (M, N), compute pairwise distances
	between all possible pairs (about 0.5 M^2) of rows in X. 
	
	Parameters
	----------
	X : 2D array of float, shape (M, N)
		Input data, M rows (features) and N columns (samples)
		
	Returns
	-------
	Xn : 1D array of float, shape (1, 0.5M^2)
	"""
	
	Xn = np.array([])
	for i in range(len(X)):
		for j in range(i + 1,len(X)):
			pair = np.array([X[i],X[j]])
			pair_dist = scipy.spatial.distance.pdist(pair,'euclidean')
			Xn = np.append(Xn,pair_dist)
	
	return Xn
	
def intergene_pairwise_distance(X, Y):
	"""for 2 input arrays of shape (M, N) each, compute pairwise distances
	between all possible pairs of rows between X and Y. 
	
	Parameters
	----------
	X, Y : 2D array of float, shape (M, N)
		Input data, M rows (features) and N columns (samples)
		
	Returns
	-------
	Z : 1D array of float, shape ((M * M), 1)
	"""
	
	Z = np.array([])
	
	for i in range(len(X)):
		for j in range(len(Y)):
			pair = np.array([X[i], Y[j]])
			pair_dist = scipy.spatial.distance.pdist(pair, 'euclidean')
			Z = np.append(Z, pair_dist)
			
	return Z
	
def RGB_split(img):
	"""Splits a given np array of an image into RGB channels

	
	Parameters
	----------
	img : 3D array of float, shape (l, w, 4)
		Input image of length l, width w and a final dimension 4, for RGBA
		
	Returns
	-------
	ch0, ch1, ch2 : 3D arrays of float, shape (l, w, 4)
		Respective RGB channels. 
	"""
	
	
	ch0 = np.zeros_like(img)
	ch1 = np.zeros_like(img)
	ch2 = np.zeros_like(img)
	
	ch0[:,:,0] = img[:,:,0]
	ch1[:,:,1] = img[:,:,1]
	ch2[:,:,2] = img[:,:,2]
	
	return ch0, ch1, ch2
	


def patch_and_Gnormalize(img, patch_size, verbose=True):
    """1. Extract patches from an image
    2. Cast to a 2D matrix 
    3. normalize the patches like normal r.v.s
    
    Make sure that dtype(img) is float, so that normalization makes sense!
    
    Parameters
	----------
	img: 3D array of float, shape (M, N, C)
		Input data, M rows, N columns, C colour channels
		
	patch_size: 2D tuple, shape (patch_width, patch_height)
		Usually (10, 10)
		
	verbose: controls verbosity. True by default.
		
	Returns
	-------
	patches: 2D array of float, shape (M2, N2)
		M2 ~= M*N, N2 = patch_width * patch_height * C
		M2 will likely be less than M * N because of how extract_patches_2d works
    """
    t0 = time.time()
    if verbose:
        print('image shape = %s' % (img.shape,))
    
	#1. Extract patches
    patches = extract_patches_2d(img, patch_size)
    #2. Cast into 2D
    patches = patches.reshape(patches.shape[0], -1)
    #3. GNormalize
    
    t1 = time.time()
    patch_size = (10,10)
    data = extract_patches_2d(img,patch_size)
    dt = time.time() - t1
    if verbose:
        print('Extracted patches. Current shape = %s' % (data.shape,))
    
    t1 = time.time()
    data = data.reshape(data.shape[0],-1)
    if verbose:
        print('Reshaped patches. Current shape = %s' % (data.shape,))


    data -= np.mean(data, axis = 0)
    data /= np.std(data, axis = 0)
    data[np.isnan(data)] = 0.0
    
    dt = time.time() - t0
    if verbose:
        print('Total time elapsed = %.3fs.' % (time.time() - t0))
        
    return patches


def sparse_codifier(y,D, transform_algo = 'omp', transform_n_nonzero_coefs = 2):
    """
    Encodes an input vector y into an output sparse vector x, 
    based on a given dictionary D
    Retired, to delete.
    """
    print('y.shape = %s ' % (y.shape,))
    print('Dictionary shape = %s' % (D.shape,))
    coder = SparseCoder(dictionary = D, transform_algorithm = transform_algo)
    x = coder.transform(y)
    print('x.shape = %s' % (x.shape,))
    
    return x


def show_dictionary(V):
	"""Might be defunct, delete if necessary"""
	
	W=(V-V.min())/(V.max() - V.min())
	
	plt.figure(figsize = (8.4, 8))
	for i, comp in enumerate(W[:100]):
		plt.subplot(10, 10, i+1)
		plt.imshow(comp.reshape(10, 10, 3), interpolation = 'nearest')
		plt.xticks(())
		plt.yticks(())
		plt.suptitle('Dictionary learnt form img')
		plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


def show_with_diff(original, reconstruction, title, cmap=plt.cm.gray):
    """Displays 3 subplots: the original image image, the reconstruction, 
    and their difference. 
    Adapted from: 
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_image_denoising.html
    
    Parameters
	----------
	original: np array of float
	    Original image
		
	reconstruction: np array of float
	    Reconstructed image. Or any other image, really.
	
	title: string
	    Do I really need to explain what a title is?
	
	cmap: string
		Colour map argument to be passed to imshow(). 
    """
    #Subplot 1: Show original
    plt.figure(figsize=(10, 6.6))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    #plt.imshow(image, vmin=0, vmax=1, interpolation='nearest')
    plt.imshow(original, cmap=cmap, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    
    #Subplot 2: Show reconstruction
    plt.subplot(1, 3, 2)
    plt.title('Reconstruction')
    #plt.imshow(image, vmin=0, vmax=1, interpolation='nearest')
    plt.imshow(reconstruction, cmap=cmap, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    
    #Subplot 3: show diff
    plt.subplot(1, 3, 3)
    difference = original - reconstruction
    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    #plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr, interpolation='nearest')
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=cmap, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)

def reconstruct_patches(x, D, patch_width, patch_height, ch):
    """
    Given a sparse matrix x, and a dictionary D, reconstruct the original signal y:
    y ~= Dx
    
    Parameters
	----------
	x: 2D array of float, shape (M, N)
		Input data, M rows, N columns. By right, each image = 1 column.
		
	D: 2D array of float, shape (M, k)
	
	patch_width: integer
		Usually 10.
	
	patch_height: integer
		Usually 10.
	
	ch: integer
		Number of colour channels, usually 3.
		
	Returns
	-------
	y: 2D array of float, shape (k, N)
		Sparse array.
    """
    
    #1. Matrix multiplication
    y = np.dot(x, D)
    print("Dot product y.shape = %s" % (patches.shape,))
    #2. Fatten the 2D matrix into a 4D block...(why 4d?)
    y = y.reshape(len(x), patch_width,patch_height,ch)
    print("Reshaped y.shape = %s" % (y.shape,))
    #Should be (692223, 10, 10, 3)
    
    return patches
    
def flatten_sparse_matrix(x, onehot = True):
    """Sets all nonzero entries of x to 1,
    then compresses to a 1D vector by taking the sum across rows
    i.e. squishes a rectangular matrix vertically.
    
    Params
    ------
    x: Array of float, shape (m, n)
    onehot: if true, converts any nonzero entry into 1
    
    Returns
    -------
    x_vec: 1D array of float, shape (n,)"""
    
    x_vec = x
    #Set all the nonzero elements of x to 1
    if onehot:
        x_vec[x_vec != 0] = 1
    x_vec = np.sum(x_vec, axis = 0)
    
    return x_vec