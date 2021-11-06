""" filescores2scs - compute mean-IoU over the support cognizance tier of TOSS

This script contains the SCSScore class. Objects of this class can be used to compute
the weighted mean intersection-over-union scores for reporting the SCS score as described 
in the paper.

A single line  in the test-split files look like this:
```
    1_2010_004917 1_2010_004917 10 1
```
The first two items are the query and support files to be used for testing. 
The third item is the test-case weight index. This is used in the calls to "update()" 
for the SCSScore object. The weight_map argument is used to map these weight-indices to actual 
weights that are used in the SCS computation.


Example Usage:
```
    # create the SCS metrics object. Note, for reproducing the numbers in the paper, 
    # keep the weight_map to these exact values, or do not specify it (defaults to these values)
    scs = SCSScore(class_list=[0,1,2,3,4], weight_map={10: 4, 5: 3, 1: 1, -1: 1})

    for i, (query_images, query_masks, support_images, support_masks, class_indices, weight_indices) 
            in enumerate(test_iterator):

        # make batch prediction with the network 
        pred_query_masks = network(support_images, support_masks, query_images)

        # update the scs object with the predictions.
        batch_size = query_images.shape[0]
        for j in range(batch_size):
            cidx = class_indices[i]
            widx = weight_indices[i]
            o_mask = torch.argmax(pred_query_masks[i], dim=1) # should yield an hxw tensor
            g_mask = query_masks[i]
            scs.update( class_idx=c, case_weight_index=w[i], output=o_mask, label=g_mask)

    # get the final score after going through all the value 
    print( scs.meanIoU() )

```
"""
from functools import reduce
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


class SCSScore:
    """ Metrics for multiple classes

        Attributes
        -----------
        ms : dict (int -> Metrics)
            stores the iou metrics for each class

        Methods
        -------
        update(classidx, case_weight, output, label)
            updates the metrics with the prediction for a particular class
        meanIoU()
            compute the mean intersection over union over the clases.
    """
    def __init__(self, class_list, weight_map = dict([ (10, 4), (5, 3), (1, 1), (-1, 1)]) ):
        """ initialize object
    
        Parameters
        ----------
        class_list : list[int]
            list of classes for which the metrics are to be recorded

        weight_map : dict
            the weights to use for each instance of a test-case. the deault values are the ones we use to report
        """
        self.ms = {}
        for c in class_list:
            self.ms[c] = dict( [(w, Metrics()) for w in weight_map.keys()] )
        self.wmap = weight_map

    def update(self, class_index, case_weight_idx, output, label):
        """ updates the metrics with the prediction for a particular class
        Parameters:
        ------------
        class_index : int
            the index of the class for which this entry is to be stored
        case_weight_idx: int (10, 5, 1, or -1)
            the weight to use for the specified test-case.
        output : Tensor[h,w]
            the argmax mask from the logits produced by the network
        label: Tensor[h,w]
            the groundtruth mask 

        Return Value:
        -------------
            the intersection over union for the current prediction

        """
        return self.ms[class_index][case_weight_idx].update( output, label )
        

    def meanIoU(self):
        """ compute the mean intersection over union over all of the classes
        Return Value:
        -------------
            the weighted mean intersection over union value
        """
        summ = {}
        for cur_class in self.ms.keys():
            summ[cur_class] = 0.
            for wtype in self.ms[cur_class].keys():
                if wtype > 0:
                    summ[cur_class] += self.ms[cur_class][wtype].iou() * self.wmap[wtype]
                else:
                    summ[cur_class] += (1 - self.ms[cur_class][wtype].iou()) * self.wmap[wtype]

        miou = float(reduce(lambda x,y: x + y, summ.values()))
        miou /= len(self.ms.keys())
        miou /= float(reduce(lambda x,y: x+y, self.wmap.values()))
        return miou


if __name__ == "__main__":
    scs = SCSScore(class_list=[0,1,2,3,4])
    def rand_mask(w, h):
        x = torch.randn((w, h))
        x[x>0.5] = 1
        x[x<=0.5] = 0
        return x

    scs.update( class_index=0,  case_weight_idx=-1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    scs.update( class_index=1,  case_weight_idx=-1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    scs.update( class_index=2,  case_weight_idx=-1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    scs.update( class_index=3,  case_weight_idx=-1, output=rand_mask(512, 512), label=rand_mask(512, 512))
    scs.update( class_index=4,  case_weight_idx=-1, output=rand_mask(512, 512), label=rand_mask(512, 512))

    print(scs.meanIoU())