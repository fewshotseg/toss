#!/bin/bash

DATASET_DIR=$1

echo "creating dataset dir"
mkdir -p  $DATASET_DIR

# Download PASCAL VOC12 (2GB)
echo "downloading PASCAL VOC data"
wget -nc -P $DATASET_DIR http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Extract images, annotations, etc.
echo "extracting PASCAL VOC images"
tar -xf $DATASET_DIR/VOCtrainval_11-May-2012.tar -C $DATASET_DIR

echo "Downloading the CANet Binary Aug maps"
wget -nc -P $DATASET_DIR/VOCdevkit/VOC2012 https://github.com/icoz69/CaNet/raw/master/Binary_map_aug.zip

echo "unarchiving"
cd $DATASET_DIR/VOCdevkit/VOC2012; unzip -q Binary_map_aug.zip; cd -;


# transfor the images and mask filenames
echo "Organizing PASCAL 5i"
python3 -m pip install tqdm shutil
python3 organizepascal5i.py  $DATASET_DIR cleanup
rm $DATASET_DIR/*.tar
rm -rf $DATASET_DIR/VOCdevkit
