import utils
import os
import nibabel as nib
import numpy as np

print()
file_paths = utils.askFiles(os.getcwd(), title='Select the nifti images to crop')
output_dir = utils.askDirectory(None, title='Select the output directory')

suffixes = ['right', 'left']
dim = np.array([32,64,64])
p_dim = dim/2
center  = np.array([25,103,47,96])

crop_range = np.array([center[0]-p_dim[0],center[0]+p_dim[0],
		 			   center[1]-p_dim[1],center[1]+p_dim[1],
					   center[2]-p_dim[2],center[2]+p_dim[2],
					   center[3]-p_dim[0],center[3]+p_dim[0]], dtype=np.uint16)

for path in file_paths:
	nii = nib.load(path)
	header = nii.header
	img = np.array(nii.get_fdata())

	for suffix in suffixes:
		if suffix == suffixes[1]:
			patch_img = np.flip(np.array(img[crop_range[6]:crop_range[7],
											 crop_range[2]:crop_range[3],
											 crop_range[4]:crop_range[5]]),0)
		
		else:
			patch_img = np.array(img[crop_range[0]:crop_range[1],
				  				 	 crop_range[2]:crop_range[3],
								 	 crop_range[4]:crop_range[5]])

		#patch_nii = nib.Nifti1Image(patch_img, nii.affine, header)
		patch_nii = nib.Nifti1Image(patch_img, affine=np.eye(4))
  
		patch_path = utils.replaceDir(utils.addSufix(path, '_patch_' + suffix), output_dir)
		nib.save(patch_nii, patch_path)

		print('[INFO]Patch cropped:"{}"'.format(patch_path))
	
	print()