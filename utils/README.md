# Utils to process the imageNet dataset

## 一、select some classes from the train dataset
create_imagenet_subset.py and imagenet100_classes.txt

The data must be placed in a source folder(**src_dir**), then give your destination folder(**dst_dir**) preprocessed with:
```
python create_imagenet_subset.py src_dir dst_dir
```

## 二、process validation dataset
classify validation dataset: process_validation.py
```
python process_validation.py val_dir devkit_dir
``` 

## 三、use the ImageFolder to load dataset
load_dataset.py