"""
Whole slide image loader module
===============================

This package contains a few simple implementation for data sources that load
images for pathology data.

.. autoclass:: WholeSlideImageDataSource
.. autoclass:: WholeSlideImageClassSampler
.. autoclass:: WholeSlideImageRandomPatchExtractor

"""
import os
import numpy as np
import sys
import deepr.data_processing.data_source as data_source
from deepr.data_processing.sample_data import NamedNdarray
from scipy.ndimage import label
from scipy.ndimage.measurements import find_objects
import multiresolutionimageinterface as mir

__all__ = ("WholeSlideImageDataSource", "WholeSlideImageClassSampler",
           "WholeSlideImageRandomPatchExtractor")

class WholeSlideImageClassSampler(object):
  """Coordinate sampler for whole slide images.

  This class parses a whole slide image mask file and extracts the classes
  present the mask. It stores a low-resolution version of the mask and
  the bounding boxes of the individual class areas to allow fast sampling
  of class coordinates.

  """
  def __init__(self, msk_file, msk_level, nr_classes, lab_to_val = {}):
    """Parses the provided mask file into bounding boxes.

    The mask file is openend using :mod:`multiresolutionimageinterface`,
    after which the mask is parsed for connected class areas using
    :mod:'scipy.ndimage'. The available classes in the mask in addition to
    the bounding boxes of the individual areas are stored in member attributes.

    Args:
      msk_file (str): Path to the mask file to parse.
      msk_level (int): The resolution level at which to parse the mask. This
        level is also stored in memory.
      nr_classes (int): Total number of classes in the entire domain (the mask
        might not contain all classes.)
      lab_to_val (dict, optional): Mapping of numerical class labels to mask
        values. If not specified the mapping is assumed to be: class_label =
        mask_value + 1 (e.g. class 0 is labeled with value 1 in the mask)

    """
    self._file_pth = msk_file
    self._mask_level = msk_level
    self._mask_level_downsample = 1.
    self._nr_classes = nr_classes
    self._lab_to_val = {}
    if not lab_to_val:
      for cur_lab in range(self._nr_classes):
        self._lab_to_val[cur_lab] = cur_lab + 1
    else:
      self._lab_to_val = lab_to_val
    self._classes_in_msk = set()
    self._bounding_box_indices = {}
    self._basename = os.path.splitext(os.path.basename(self._file_pth))[1]
    reader = mir.MultiResolutionImageReader()
    label_msk = reader.open(self._file_pth)
    if label_msk:
        dims = label_msk.getLevelDimensions(self._mask_level)
        self._mask_level_downsample = label_msk.getLevelDownsample(self._mask_level)
        self._msk_img = np.squeeze(label_msk.getUCharPatch(0, 0, dims[0], dims[1], self._mask_level)[:,:,0])
        for cur_label in range(self._nr_classes):
            slices = find_objects(label(self._msk_img == self._lab_to_val[cur_label])[0])
            bbox_ind = []
            max_size = 0.0
            for s in slices:
                bbox_size = float((s[0].stop - s[0].start) * (s[1].stop - s[1].start))
                if bbox_size > max_size:
                    max_size = bbox_size
                bbox_ind.append([s[0].start, s[0].stop,
                                 s[1].start, s[1].stop, bbox_size])
            # Make size relative and sort bbox_ind on size
            for ind in range(len(bbox_ind)):
                bbox_ind[ind][4] /= max_size
            bbox_ind.sort(key=lambda x: x[4])
            if slices:
                self._classes_in_msk.add(cur_label)
                self._bounding_box_indices[cur_label] = bbox_ind
        label_msk.close()

  def getSamplerClasses(self):
    """Returns which classes where found in the mask file.

    Returns:
      list of ints: sorted list of available class labels.
    """
    return sorted(list(self._classes_in_msk))

  def getCoordinateForClass(self, class_label):
    """Returns a radom coordinate at level 0 for the requested class_label.

    Args:
      class_label (int): The class label for which a position is requested.

    Returns:
      tuple of ints: a set of coordinates if class_label is present in the mask
        else defaults to (-1, -1).
    """
    if class_label in self._classes_in_msk:
        rnd_ind = np.random.randint(self.__bisect_bbox_size(np.random.rand(), self._bounding_box_indices[class_label]), len(self._bounding_box_indices[class_label]))
        bbox_ind = self._bounding_box_indices[class_label][rnd_ind]
        loc = None
        while loc is None:
            row = np.random.randint(bbox_ind[0], bbox_ind[1])
            col = np.random.randint(bbox_ind[2], bbox_ind[3])
            if self._msk_img[row, col] == self._lab_to_val[class_label]:
                loc = (int(row * self._mask_level_downsample), int(col * self._mask_level_downsample))
                return loc
    return (-1, -1)

  def __bisect_bbox_size(self, random_nr, bbox_list):
      lo = 0
      hi = len(bbox_list) - 1
      while lo < hi:
             mid = (lo + hi) // 2
             if bbox_list[mid][4] < random_nr: lo = mid + 1
             else: hi = mid
      return lo

class WholeSlideImageRandomPatchExtractor(data_source.UncachedDataSource):
  """Data source to obtain ranom samples from multiple WSIs.

  This class combines a list of :class:`WholeSlideDataSources` with their
  corresponding :class:`WholeSlideImageClassSampler`. This DataSource samples
  randomly and thus has its length defined as the largest possible 64bit
  integer number.

  """
  def __init__(self, wsi_data_sources, class_samplers, mask_scale=1.):
    """Identifies classes in the lists of sources and samplers.

    Args:
      wsi_data_sources (list of WholeSlideDataSources): The data sources to
        get samples from.
      class_samplers (list of WholeSlideImageClassSampler): The samplers
        corresponding to the WholeSlideDataSources. The list has to be the
        same length as wsi_data_sources and the indices in the lists should
        be matched (e.g. index 0 in class_samplers should be the sampler for
        the WholeSlideDataSources at index 0 in wsi_data_sources).
      mask_scale (float, optional): If the masks were stored at a different
        resolution than the original whole slide images, this scaling factor
        can be used to correct the coordinates.
    """
    super(WholeSlideImageRandomPatchExtractor, self).__init__([])
    self._wsi_data_sources = wsi_data_sources
    self._class_samplers = class_samplers
    self._classes_per_sampler = {}
    self._mask_scale = mask_scale
    for i, (original, sampler) in enumerate(zip(self._wsi_data_sources, self._class_samplers)):
      available_classes = sampler.getSamplerClasses()
      for av_class in available_classes:
        if not av_class in self._classes_per_sampler.keys():
          self._classes_per_sampler[av_class] = []
        self._classes_per_sampler[av_class].append((original, sampler))

  def __len__(self):
    """Returns the largest 64 bit integer as this DataSources samples randomly.
    """
    return sys.maxsize

  def _process(self, index, subdata_selection=data_source.DataSource.PrefetchFlags.IMAGE_DATA, *sub_data_source_data):
    """Returns a random sample with a single input and a single label.

    This functions randomly selects a class from the list of available classes
    and subsequently selects a random whole slide file to sample this class
    from. A coordinate is obtained from the WholeSlideImageClassSampler and a
    patch extracted from the WholeSlideDataSources. These are combined into a
    Sample, which is returned.

    Args:
      index (int): the index within the DataSource to sample from. In this
        derived class this is not used as the samples are extracted randomly.
      subdate_selection (enum): indicates how subdata is prefetched. See
        :class:`DataSource`
      sub_data_source_data (dict of Samples): data from previous DataSources
        up the pipeline.

    Returns:
      :class:`Sample` A sample with one input (patch of image data) and one
        label (the class label).
    """
    class_label = self._classes_per_sampler.keys()[np.random.randint(0, len(self._classes_per_sampler.keys()))]
    original, sampler = self._classes_per_sampler[class_label][np.random.choice(len(self._classes_per_sampler[class_label]))]
    index = None
    tries = 0
    while not index:
      coord = sampler.getCoordinateForClass(class_label)
      try:
        index = original._level_position_to_index(int(coord[1] * self._mask_scale), int(coord[0] * self._mask_scale))
      except :
        index = None
        print "???",sys.exc_info()[0]
      tries += 1
      if tries == 1000:
        raise ValueError("Could not sample position for class after 1000 tries")
    sample = original[index]
    label_result = NamedNdarray("label")
    label_result.data = np.array(class_label)
    sample.labels = (label_result,)
    return sample

class WholeSlideImageDataSource(data_source.UncachedDataSource):
  """Data source for whole slide pathology images.

  Given a whole slide image file, this DataSource provides indexed access to
  all valid pixel data for a given tile size at a given resolution level. Care
  is taken that tiles which are (partly) outside of the image cannot be
  requested. When indexing the image a :class:`Sample` is returned with one
  input (a tile of image data) and no labels.
  """
  def __init__(self, file_path, tile_size, level, requested_data_type=mir.InvalidDataType):
    """Parses a whole slide image file and sets internal variables.

    Args:
      file_path (str): the path to the whole slide image file.
      tile_size (list of ints): the dimensionality of the requested tiles of
        image data.
      level (int): resolution level at which to obtain the requested tiles.
      requested_data_type (enum, optional): if the user would like to obtain
        the tiles in a different datatype than the one in which they are stored
        this flag can be changed.
    """
    super(WholeSlideImageDataSource, self).__init__([])
    self._file_path = file_path

    reader = mir.MultiResolutionImageReader()
    self._wsi = reader.open(self._file_path)
    if self._wsi :
        self._nr_of_levels = self._wsi.getNumberOfLevels()
        self._level = level
        if self._level > self._nr_of_levels:
          raise IndexError("Specified level exceeds levels in file.")
        self._level0_dimensions = self._wsi.getLevelDimensions(0)
        self._level_dimensions = self._wsi.getLevelDimensions(self._level)
        self._tile_size = tile_size
        if self._level_dimensions[0] < self._tile_size[0] or self._level_dimensions[1] < self._tile_size[1]:
          raise ValueError("Specified tile size is smaller than requested level size.")
        self._level_downsample = self._wsi.getLevelDownsample(self._level)
        self._wsi_datatype = mir.InvalidDataType
        self._requested_data_type = requested_data_type
        self._wsi_datatype = self._wsi.getDataType()
        self._level0_tile_size = [int(np.ceil(x * self._level_downsample)) for x in self._tile_size]
        self._useable_width = int(self._level0_dimensions[0] - self._level0_tile_size[0] + 1)
        self._useable_height = int(self._level0_dimensions[1] - self._level0_tile_size[1] + 1)
        self._max_index = self._useable_width * self._useable_height

  def __len__(self):
    """Returns the length based on the requested level and requested tile size.
    """
    return self._max_index

  def _index_to_level_position(self, index):
    """ Converts the provided index to the corresponding image position.

    Args:
      index (int): the requested index

    Returns:
      tuple of ints: the corresponding image position (x, y) of the center of
        the requested tile at index.
    """
    if index >= self._max_index or index < 0:
        raise IndexError("Index out of range")
    y_pos_topleft = int(np.floor(index / self._useable_width))
    x_pos_topleft = int(np.floor(index - y_pos_topleft * self._useable_width))
    y_pos_center = int(y_pos_topleft + self._level0_tile_size[0] / 2.)
    x_pos_center = int(x_pos_topleft + self._level0_tile_size[1] / 2.)
    return (x_pos_center, y_pos_center)

  def _level_position_to_index(self, x_pos, y_pos):
    """Return the corresponding index for a certain position.

    Args:
      x_pos (int): x-position of the center of a tile.
      y_pos (int): y-position of the center of a tile.

    Returns:
      int: index corresponding to the tile with x_pos, y_pos at its center.
    """
    y_pos_topleft = int(y_pos - self._level0_tile_size[0] / 2.)
    x_pos_topleft = int(x_pos - self._level0_tile_size[1] / 2.)

    if y_pos < 0 or y_pos >= self._level0_dimensions[1]:
      raise IndexError("Y-position outside of image bounds")

    if x_pos < 0 or x_pos >= self._level0_dimensions[0]:
      raise IndexError("X-position outside of image bounds")

    if y_pos_topleft < 0 or y_pos_topleft >= self._level0_dimensions[1] or x_pos_topleft < 0 or x_pos_topleft >= self._level0_dimensions[0]:
      raise IndexError("Patch requested which is partially outside the image")

    index = y_pos_topleft * self._useable_width + x_pos_topleft
    if index >= self._max_index or index < 0:
        raise IndexError("Index out of range for position " + str((x_pos, y_pos)))
    return int(index)

  def getSampleAt(self, x_pos, y_pos):
    """Convenience function to directly get the tile with center at (x, y).

    Args:
      x_pos (int): x-position of the center of a tile.
      y_pos (int): y-position of the center of a tile.c
    """
    return self._process(self._level_position_to_index(x_pos, y_pos))

  def _process(self, index, subdata_selection=data_source.DataSource.PrefetchFlags.IMAGE_DATA, *sub_data_source_data):
    """Returns a sample with one input and no labels.

    Given an index this returns the corresponding tile with a width and height
    of tile_size as a :class:`Sample`. The datatype of the image data is
    determined by requested_data_type.

    Args:
      index (int): the index within the DataSource to sample from.
      subdate_selection (enum): indicates how subdata is prefetched. See
        :class:`DataSource`
      sub_data_source_data (dict of Samples): data from previous DataSources
        up the pipeline.

    Returns:
      :class:`Sample` A sample with one input (patch of image data) and no
        label
    """
    x_pos_center, y_pos_center = self._index_to_level_position(index)
    y_pos_topleft = int(y_pos_center - self._level0_tile_size[0] / 2.)
    x_pos_topleft = int(x_pos_center - self._level0_tile_size[1] / 2.)
    requested_data_type = self._wsi_datatype
    if self._requested_data_type != mir.InvalidDataType:
        requested_data_type = self._requested_data_type
    img_data = None
    if requested_data_type == mir.UChar:
        img_data = self._wsi.getUCharPatch(x_pos_topleft, y_pos_topleft, self._tile_size[0], self._tile_size[1], self._level)
    elif requested_data_type == mir.UInt16:
        img_data = self._wsi.getUInt16Patch(x_pos_topleft, y_pos_topleft, self._tile_size[0], self._tile_size[1], self._level)
    elif requested_data_type == mir.UInt32:
        img_data = self._wsi.getUInt32Patch(x_pos_topleft, y_pos_topleft, self._tile_size[0], self._tile_size[1], self._level)
    elif requested_data_type == mir.Float:
        img_data = self._wsi.getFloatPatch(x_pos_topleft, y_pos_topleft, self._tile_size[0], self._tile_size[1], self._level)
    image_result = NamedNdarray("image")
    image_result.data = img_data.transpose(2, 0, 1)
    result = data_source.Sample()
    result.inputs = (image_result,)
    return result
