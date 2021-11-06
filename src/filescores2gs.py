""" filescores2gs - compute meaniou over the generalization tier of TOSS

This script contains the ClasswiseMetrics class. Objects of this class can be used to compute
the mean intersection-over-union scores on a per-prediction basis for each test class.
"""
import torch

class Metrics:
    """ Metrics for a single class

        Attributes
        -----------
        intersection : float
            sum total of all mask intersections between the predicted and the ground-truth masks
        union : float
            sum total of all white pixel counts across all the masks
        eps : float
            epsilon value used for divisions to avoid div-by-zero erros

        Methods
        -------
        update(output, label)
            updates the sum of intersection and union values
        iou()
            compute the overall intersection over union
    """

    def __init__(self, epsilon=1e-7):
        """ initialize object.
        Parameters:
        ------------
        epsilon : float
            epsilon value used for divisions to avoid div-by-zero erros

        """
        self.intersection = 0
        self.union = 0
        self.eps = epsilon

    def _i_and_u(self, output, label):
        t = output[label == 1]
        i = len(t[t == 1])
        t = output + label
        u = len(t[t > 0])
        return i, u

    def update(self, output, label):
        """ update intersection and union records.

        Parameters:
        ------------
        output : Tensor[h,w]
            the argmax mask from the logits produced by the network
        label: Tensor[h,w]
            the groundtruth mask 

        Return Value:
        -------------
            the intersection over union for the current prediction

        """
        i, u = self._i_and_u(output, label)
        assert((i >= 0) and (u >= 0))
        self.intersection += i
        self.union += u
        return float(i) / (u + self.eps)

    def iou(self):
        """ get the overall intersection-over-union values

        Parameters
        -----------
            None
        
        Return Value:
        -------------
            the overall intersection over union for the collected predictions
        """
        return float(self.intersection) / (self.union + self.eps)


class ClasswiseMetrics:
    """ Metrics for multiple classes

        Attributes
        -----------
        ms : dict (str -> Metrics)
            stores the iou metrics for each class

        Methods
        -------
        update(class_index, output, label)
            updates the metrics with the prediction for a particular class
        meanIoU()
            compute the mean intersection over union over the clases.
    """
    def __init__(self):
        """ initialize object"""
        self.ms = {}

    def update(self, class_index, output, label):
        """ updates the metrics with the prediction for a particular class
        Parameters:
        ------------
        class_index : int
            the index of the class for which this entry is to be stored
        output : Tensor[h,w]
            the argmax mask from the logits produced by the network
        label: Tensor[h,w]
            the groundtruth mask 

        Return Value:
        -------------
            the intersection over union for the current prediction

        """
        if not(class_index in self.ms):
            self.ms[class_index] = Metrics()
        return self.ms[class_index].update(output, label)

    def meanIoU(self):
        """ compute the mean intersection over union over all of the classes
        Return Value:
        -------------
            the mean intersection over union value
        """
        summ = 0.0
        for cur_class in self.ms.keys():
            summ += float(self.ms[cur_class].iou())
        return summ/len(self.ms.keys())


if __name__ == "__main__":
    cm = ClasswiseMetrics()
    def rand_mask(w, h):
        x = torch.randn((w, h))
        x[x>0.5] = 1
        x[x<=0.5] = 0
        return x

    cm.update( class_index=0, output=rand_mask(512, 512), label=rand_mask(512, 512))
    cm.update( class_index=1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    cm.update( class_index=2, output=rand_mask(512, 512), label=rand_mask(512, 512))
    cm.update( class_index=3, output=rand_mask(512, 512), label=rand_mask(512, 512))
    cm.update( class_index=4, output=rand_mask(512, 512), label=rand_mask(512, 512))

    print(cm.meanIoU())