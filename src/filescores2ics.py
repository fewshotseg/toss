""" filescores2ics - compute mean-iou over the query complexity tier of TOSS

This script contains the TestSetQCS class. Objects of this class can be used to compute
the low-complexity accuracy and high complexity accuracy scores for the entire test dataset

The scores for each fold must be supplied for each sub-partition (easy-salient, easy-non-salient, hard-salient and
hard-non-salient) using the update function. The final LCA and HCA can be obtained at a single fold-level using the 
fold_lca() and fold_hca() functions. Average LCA and HCA across the folds can be obtained using the mean_lca() and 
mean_hca() functions.

Example:
--------
```
    tsq = TestSetQCS()
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_EasySalient, mean_iou=0.5)
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_EasyNonSalient, mean_iou=0.35)
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_HardSalient, mean_iou=0.38)
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_HardNonSalient, mean_iou=0.13)

    print(tsq.fold_lca(0))
    print(tsq.fold_hca(0))
```
"""

from functools import reduce

class TestSetQCS:
    """ Weighted mean iou computation for the query complexity tier of TSS
        
        Attributes
        -----------
        fold_mious : [ [float] ] 
            mean intersection over union values for each fold, for each partition

        Methods
        -------
        update( fold_number, part_type, mean_iou )
            updates the mean intersection over union values for the given fold, and partition
        fold_lca(fold_number)
            the low-complexity accuracy for the given fold-number
        fold_hca(fold_number)
            the high-complexity accuracy for the given fold-number
        mean_lca()
            the low-complexity accuracy across the 4-folds
        mean_hca()
            the high-complexity accuracy across the 4-folds
    """

    # Constants
    PartType_EasySalient= 0         # Easy-Salient
    PartType_EasyNonSalient= 1      # Easy-Non-salient
    PartType_HardSalient = 2        # Hard-Salient
    PartType_HardNonSalient = 3     # Hard-Non-Salient


    def __init__(self):
        """ initialize object """
        self.fold_mious = [[0. for _ in range(4)] for _ in range(4)]

    def update(self, fold_number, part_type, mean_iou):
        """ update intersection and union records.

        Parameters:
        ------------
        fold_number : int
            the argmax mask from the logits produced by the network
        part_type: TestSetQCS.(PartType_EasySalient|PartType_EasyNonSalient|PartType_HardSalient|PartType_HardNonSalient)
            the type of the partition
        mean_iou : float
            the value of the mean intersection-over-union

        Return Value:
        -------------
            None
        """
        self.fold_mious[fold_number][part_type] = mean_iou

    def _weighted_average(self, fold_index, iorder):
        f = self.fold_mious[fold_index]
        return  (3 * f[iorder[0]] + 0.75 * f[iorder[1]] + 0.75 * f[iorder[2]] + 0.5 * f[iorder[3]]) / 5

    def fold_lca(self, fold_number):
        """ get the low-complexity accuracy for the given fold

        Parameters
        -----------
            fold_number : int
                the fold for which the score is required.
        
        Return Value:
        -------------
            the LCA value for the specified fold
        """
        return self._weighted_average(
            fold_index=fold_number, 
            iorder=[
                TestSetQCS.PartType_EasySalient,
                TestSetQCS.PartType_EasyNonSalient,
                TestSetQCS.PartType_HardSalient,
                TestSetQCS.PartType_HardNonSalient
            ])

    def fold_hca(self, fold_number):
        """ get the high-complexity accuracy for the given fold

        Parameters
        -----------
            fold_number : int
                the fold for which the score is required.
        
        Return Value:
        -------------
            the HCA value for the specified fold
        """
        return self._weighted_average(
            fold_index=fold_number, 
            iorder=[
                TestSetQCS.PartType_HardNonSalient,
                TestSetQCS.PartType_HardSalient,
                TestSetQCS.PartType_EasyNonSalient,
                TestSetQCS.PartType_EasySalient,
            ])

    def mean_lca(self):
        """ get the low-complexity accuracy for all the folds

        Parameters
        -----------
            None
        
        Return Value:
        -------------
            the LCA averaged for all the folds
        """
        return reduce(lambda x,y: x+y, [self.fold_lca(fold_number) for fold_number in range(4)])/4.


    def mean_hca(self):
        """ get the high-complexity accuracy for all the folds

        Parameters
        -----------
            None
        
        Return Value:
        -------------
            the HCA value for all the folds
        """
        return reduce(lambda x,y: x+y, [self.fold_hca(fold_number) for fold_number in range(4)])/4.


if __name__ == "__main__":
    tsq = TestSetQCS()
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_EasySalient, mean_iou=0.5)
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_EasyNonSalient, mean_iou=0.35)
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_HardSalient, mean_iou=0.38)
    tsq.update(fold_number=0, part_type=TestSetQCS.PartType_HardNonSalient, mean_iou=0.13)

    print(tsq.fold_lca(0))
    print(tsq.fold_hca(0))