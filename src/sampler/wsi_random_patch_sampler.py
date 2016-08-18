from __future__ import division
import numpy as np
import multiresolutionimageinterface as mir
import wsi_sampler
#import random

class WSIRandomPatchSampler(object):

    def __init__(self, wsi_samplers, labels=[1,2,3]):

        assert len(labels) > 0
        assert len(wsi_samplers) > 0

        self.labels = labels
        self.per_label_sampler_list = {c:[] for c in labels}
        self.per_label_weight_list = {c:[] for c in labels}

        # Work out which wsi_images contain which of the labels
        for wsi_sampler in wsi_samplers:
            for label in wsi_sampler.labels:

                if label not in labels:
                    print "Warning, unknown label", label, "in wsi_samplers list!"
                    continue #Maybe actually warn instead?

                self.per_label_sampler_list[label].append(wsi_sampler)
                self.per_label_weight_list[label].append(wsi_sampler.mask.volume)

        # For every labeler determine the volume (amount of true pixels for label)
        # weigh the probability of choosing that image by this value
        for l in labels:
            m = sum(self.per_label_weight_list[l])
            self.per_label_weight_list[l] = map(lambda x: x/m, self.per_label_weight_list[l])

            # At least one sampler (=image) per label
            assert len(self.per_label_sampler_list[l]) > 0

    def sample_label(self, label):
        # Select a random file
        sampler = np.random.choice(self.per_label_sampler_list[label], p=self.per_label_weight_list[label])

        # Sample an image from it
        image = sampler.sample(label)

        return image, sampler.filename

    #Sample N images, randomly select labels uniformly
    def sample_n_balanced(self, n):
        labels = np.random.choice(self.labels, n)

        samples = [self.sample_label(l) for l in labels]
        images, filenames = zip(*samples)
        return np.array(images), labels, filenames
        

    