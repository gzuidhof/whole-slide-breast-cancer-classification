from deepr.data_processing.wsi_data_sources import WholeSlideImageDataSource, WholeSlideImageClassSampler, WholeSlideImageRandomPatchExtractor
from deepr.data_processing.simple_operations import LambdaVoxelOperation
from deepr.data_processing.batch_generator import RandomBatchGenerator
from deepr.data_processing.batch_adapter import BatchAdapterLasagne
from deepr.data_processing.simple_operations import OrdinalLabelVectorizer
import util

nr_classes=3
#labels_dict = {0:1, 1:2, 2:3}
labels_dict = {q:q+1 for q in range(nr_classes)}


def prepare_lasagne_patch(random_train_items, msk_src, network_parameters):
    print "getting all masks"
    tra_wsi = []
    tra_msk = []
    for tra_fl in random_train_items: # 20X
        tra_wsi.append(WholeSlideImageDataSource(tra_fl, (network_parameters.image_size, network_parameters.image_size), network_parameters.data_level))
        tra_msk.append(WholeSlideImageClassSampler(msk_src[tra_fl], 0, nr_classes, labels_dict))

    patch_extractor = WholeSlideImageRandomPatchExtractor(tra_wsi, tra_msk)
    train_data_source = LambdaVoxelOperation(patch_extractor, name = "image normalizer",
                                 input_names = ["image"],
                                 label_names = [],
                                 function = util.normalize_image)

    final_data_source = OrdinalLabelVectorizer(train_data_source, "label", "label", nr_classes)
    batch_generator = RandomBatchGenerator([final_data_source])
    batch_generator_lasagne = BatchAdapterLasagne(batch_generator)
    batch_generator_lasagne.select_inputs(["image"])
    batch_generator_lasagne.select_labels(["label"])
    print "... is done"
    return batch_generator_lasagne
