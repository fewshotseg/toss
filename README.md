# The TOSS Dataset
Supplementary to our work on the Neurips 2022 Data and Benchmarks Track submission located [here](https://openreview.net/pdf?id=BlcUQYxknbX).


## Dataset Details
1. [Dataset Design choices](https://github.com/fewshotseg/toss/blob/main/docs/DatasetDesign.md)
2. [Setting Up for Benchmarking](https://github.com/fewshotseg/toss/blob/main/docs/Setup.md)
3. [Supplemental information on Network biases](https://github.com/fewshotseg/toss/blob/main/BiasesMoreInfo.md) including COCO-20<sup>i</sup> performance for salience-guided training set selection.


## Dataset organization
TOSS is a re-organization of the images and masks provided by Shaban et al. ([PASCAL 5<sup>i</sup>](https://www.cc.gatech.edu/~bboots3/files/OneShotSegmentation.pdf)) and by Li et al. ([FSS-1000](https://github.com/HKUSTCV/FSS-1000)). The reorganization instructions are included in the [setup](https://github.com/fewshotseg/toss/blob/main/docs/Setup.md). Essentially there are three contributions:
1. PASCAL 5<sup>i</sup> and COCO 20<sup>i</sup> fixed test splits - randomly sampled, specified pairs that allow for uniform testing.
2. The TOSS test splits for nuanced evaluation
3. Evaluation metrics.

### Fixed Test Splits
PASCAL 5<sup>i</sup> - located in data/fixedsplits/pascal5i - each test file has 15000 pairs (3000 per test class, (query, support)).
COCO 20<supp>i<sup> - located in data/fixedsplits/coco20i - each test file has at least 9000 pairs( query, support)


### TOSS test splits
There are three tiers of splits - 
1. The attribute-oriented splits that evaluates a model for different complexities of images. This is in data/tiers/attribute. Each file is named splitN\_q\_[easy|hard]\_[sal|nsal].txt. N is the PASCAL-5<sup>i</sup> fold (0-based). Each file has at least 15000 pairs.
2. The support-cognizance splits are in data/suppcog. The files are named splitN\_tier2.txt. Again N is for the PASCAL-5<sup>i</sup> split. 
3. The generalization testt file is in the data/general/data/general\_tier.txt. It has 5006 pairs of files to be used as query and support. 
