# Segmentation Code

I've made some headway in the segmentation code. I've effectively copied much of the U-Net TGS Blob detection code and am using that for starting. I've changed activation from elu to relu and have changed datatypes and size of certain arrays used in code to properly match the data. I've gotten it properly to load and convolve all neccessary data, but still need to add code to perform analysis. Currently Adam Conrad's hard_dice metric is being used. I've also separated the data in Samos such that 20 images are being used for training and 10 images are being used for testing. Take a look, as I'm hoping to generate some basic results from this code over the weekend. Let me know what you think and if there is anything that needs to be changed.


-----------------------------------------------------------

# Automatic Display Code

Below is a set of Automatic Display code using 

CHASSSYMM3_to_ABA.xlsx 
B51315_T1_masked.nii 
B51315_invivoAPOE1_labels.nii.gz 
The Python Script described below 

When putting each of the aforementioned files into a single folder (can be called anything but I’m going to use the name “demo” for simplification purposes). 

## The Code Below does the following in printed order:

1. Converts the labels from the xlsx file into a compatible “info” json file for display of labels, ids, and descriptions

* NOTE: This does NOT use neuroglancers-scipts python module for xlsx accessibility. If you want me to instead use neuroglancers-scipts to create the info files, let me know as this may be more stable

2. Converts the segmentation nii.gz file to unsigned short from short or other format

* If a float32 is used, or any other non-unsigned integer based segmentation file, multilabel display does not work properly without this conversion step. This step should hopefully allow for anything we want to be displayed to appear properly

* I have yet to encounter a file-type that has been adversely affected by the conversion; everything appears to be ok when I do it, but I would be cautious about this statement since my testing of this script has been more limited

3. Creates the URL necessary for display of files

* Several assumptions are made with respect to the URL :  Voxel size is assumed to be 0.0001m in all three dimensions. This can be accounted for by providing voxel size directly, but cannot be assumed. It is also relatively straightforward for users to change this when the image is displayed, so I have chosen to not focus on this specifically.

* The orientation of the display is specifically set from a previous URL. I believe this can be assumed without providing a specific orientation, but there is no specific benefit to setting it to anything other than its current setting 


Means of calling code from Terminal
See notes in Python script for additional rules for usage
```python3 load_multi.py CHASSSYMM3_to_ABA.xlsx B51315_T1_masked.nii B51315_invivoAPOE1_labels.nii.gz http://localhost:8888/brainwhiz/demo/ http://localhost:8888/brainwhiz/gallery/neuroglancer1/src/neuroglancer/```
