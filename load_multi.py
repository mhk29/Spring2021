import nibabel as nib
import numpy as np
from skimage.util import img_as_uint
import glob, os, sys
import xlrd
import webbrowser

info = sys.argv[1] # so any input file can be used
pic = sys.argv[2] # so any image can be used
file = sys.argv[3] # so any segmentation file can be used
pw = sys.argv[4] # pathway is properly understood
ng = sys.argv[5] # neuroglancer pathway 

# for usefulness in our specific case, specific inputs are used
# for a more general usage of this script, replace the files used for
# wb, file, and pic with command line inputs
# ALSO! Make sure everything is in one folder for inputs (demo in this case)

# example command line
# make sure 
# me@mycomputer demo % python3 load_multi.py 
# 								CHASSSYMM3_to_ABA.xlsx 
# 								B51315_T1_masked.nii 
# 								B51315_invivoAPOE1_labels.nii.gz 
# 								http://localhost:8888/brainwhiz/demo/ 
# 								http://localhost:8888/brainwhiz/gallery/neuroglancer1/src/neuroglancer/

# repeated on one line below for convenience:
# python3 load_multi.py CHASSSYMM3_to_ABA.xlsx B51315_T1_masked.nii B51315_invivoAPOE1_labels.nii.gz http://localhost:8888/brainwhiz/demo/ http://localhost:8888/brainwhiz/gallery/neuroglancer1/src/neuroglancer/

# wb = xlrd.open_workbook("./../upload/CHASSSYMM3_to_ABA.xlsx") # change to absolute path for safety
wb = xlrd.open_workbook("./"+info) # toggle with above 
sheet = wb.sheet_by_index(0)

a = '{"@type": "neuroglancer_segment_properties","inline": {"ids":['
b = '],"properties": [{"id": "label","type": "label","values":['
c = ']},{"id": "description","type": "description","values": ['
d = ']}]}}'

for i in range(166): # 166 has to be hard coded since there are other rows below in input xlsx
	a = a + '"' + str(int(sheet.cell_value(i+1,2))) + '"'
	b = b + '"' + str(sheet.cell_value(i+1,1)) + '"'
	c = c + '"' + str(sheet.cell_value(i+1,0)) + '"'
	if i < 165:
		a = a + ','
		b = b + ','
		c = c + ','

a = a + ','
b = b + ','
c = c + ','

for i in range(166): # 166 has to be hard coded since there are other rows below in input xlsx
	a = a + '"' + str(int(sheet.cell_value(i+1,3))) + '"'
	b = b + '"' + str(sheet.cell_value(i+1,1)) + '"'
	c = c + '"' + str(sheet.cell_value(i+1,0)) + '"'
	if i < 165:
		a = a + ','
		b = b + ','
		c = c + ','

out = a + b + c + d

# json = open("./../upload/info", "w") # info is name of file, NOT info.json # toggle on and off
json = open("./info","w")
json.write(out)
json.close()

# file = "./../gallery/ss/B51315_invivoAPOE1_labels.nii.gz" # toggle on and off
image = nib.load(file)

# to be extra sure of not overwriting data:
new_data = np.copy(image.get_fdata())
hd = image.header
print(hd)

# in case you want to remove nan:
new_data = np.nan_to_num(new_data)
new_data = new_data.astype(hd.get_data_dtype())
# convert to unsigned short (uint16) 
# new_data = (65539.0000459/65536) * img_as_uint(new_data)
new_data = img_as_uint(new_data)


# update data type:
new_dtype = np.uint16  # for example to cast to int8 or uint16.
new_data = new_data.astype(new_dtype)
image.set_data_dtype(new_dtype)

# if nifty1
if hd['sizeof_hdr'] == 348:
    new_image = nib.Nifti1Image(new_data, image.affine, header=hd)
# if nifty2
elif hd['sizeof_hdr'] == 540:
    new_image = nib.Nifti2Image(new_data, image.affine, header=hd)
else:
    raise IOError('Input image header problem')

print(new_image)

nib.save(new_image, os.path.splitext(file)[0]+"_new.nii")

# url = 'http://localhost:8888/brainwhiz/gallery/neuroglancer1/src/neuroglancer/#!{"dimensions":{"x":[0.00009999999403953553,"m"],"y":[0.00010000000149011611,"m"],"z":[0.00010000000149011611,"m"]},"position":[120.20738220214844,101.54607391357422,56.297542572021484],"crossSectionScale":0.651266211239701,"projectionOrientation":[0.06507657468318939,0.9975184798240662,0.026847289875149727,-0.0010944941313937306],"projectionScale":183.65973866423593,"layers":[{"type":"segmentation","source":["precomputed://http://localhost:8888/brainwhiz/upload",{"url":"nifti://http://localhost:8888/brainwhiz/gallery/ss/B51315_invivoAPOE1_labels.nii_new.nii","transform":{"matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0]],"outputDimensions":{"x":[0.00009999999403953553,"m"],"y":[0.00010000000149011611,"m"],"z":[0.00010000000149011611,"m"]}}}],"tab":"source","name":"upload"},{"type":"image","source":{"url":"nifti://http://localhost:8888/brainwhiz/gallery/RARE_masks/B51315_T1_masked.nii","transform":{"matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0]],"outputDimensions":{"x":[0.00009999999403953553,"m"],"y":[0.00010000000149011611,"m"],"z":[0.00010000000149011611,"m"]}}},"tab":"source","name":"B51315_T1_masked.nii"}],"selectedLayer":{"layer":"B51315_T1_masked.nii","visible":true},"layout":"4panel","partialViewport":[0,0,1,1]}'
url = ng + '#!{"dimensions":{"x":[0.00009999999403953553,"m"],"y":[0.00010000000149011611,"m"],"z":[0.00010000000149011611,"m"]},"position":[120.20738220214844,101.54607391357422,56.297542572021484],"crossSectionScale":0.651266211239701,"projectionOrientation":[0.06507657468318939,0.9975184798240662,0.026847289875149727,-0.0010944941313937306],"projectionScale":183.65973866423593,"layers":[{"type":"segmentation","source":["precomputed://'+pw+'",{"url":"nifti://'+pw+os.path.splitext(file)[0]+'_new.nii","transform":{"matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0]],"outputDimensions":{"x":[0.00009999999403953553,"m"],"y":[0.00010000000149011611,"m"],"z":[0.00010000000149011611,"m"]}}}],"tab":"source","name":"segmentation"},{"type":"image","source":{"url":"nifti://'+pw+pic+'","transform":{"matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0]],"outputDimensions":{"x":[0.00009999999403953553,"m"],"y":[0.00010000000149011611,"m"],"z":[0.00010000000149011611,"m"]}}},"tab":"source","name":"'+pic+'"}],"selectedLayer":{"layer":"'+pic+'","visible":true},"layout":"4panel","partialViewport":[0,0,1,1]}'

print("\nIf neuroglancer does not appear, copy this link into web browser:")
print("\n" + url + "\n")
webbrowser.open_new(url)

# OK! So I sort of figured out what's wrong with this webbrowser command above
# the first instance of '#' in the URL above is often turned into '%23'
# which neuroglancer does not recognize
# this always occurs for the command below but on occassion and on different
# computers I've gotten lucky a few times with it
# I'm going to investigate this a bit and see what we can do
# I think it's an OS issue but could also be a browser issue

