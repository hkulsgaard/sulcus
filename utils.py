import tkinter as tk
from tkinter import filedialog as fd
import os

def selectFiles(folder, title=None):
# This function opens a window for multiple file seleccion

	# creates the main window
	root = tk.Tk()

	# create an instance of the open file dialog assigned to the main window
	diag = fd.Open(root)

	# set the file types to be filtered (parameterizable)
	filetypes = (('Nifti Image', ['*.nii','*.nii.gz']), ('Compressed Nifti Image', '*.nii.gz'),('All files', '*.*'))
	
	# show the file selection dialog and get the selected files
	fnames = diag.show(filetypes=filetypes, initialdir = folder, multiple=1, title=title)
	
	#destroy the main window
	root.destroy()

	return fnames

def askFiles(folder, title=None):

	filetypes = (('Nifti Image', ['*.nii','*.nii.gz']), ('Compressed Nifti Image', '*.nii.gz'),('All files', '*.*'))
	
	fnames = fd.askopenfilenames(filetypes=filetypes, initialdir = folder, title=title)
	
	return fnames

def askConfigFile(folder, title=None):

	filetypes = (('YAML', ['*.yaml']),('All files', '*.*'))
	
	fnames = fd.askopenfilename(filetypes=filetypes, initialdir = folder, title=title)
	
	return fnames

def askDirectory(folder, title=None):
	path = fd.askdirectory(initialdir=None, title=title)
	return path

def addSufix(path, suffix, ext=None):
# This function takes a file path and adds a suffix to the file name
	fpath = os.path.split(path)
	fname = os.path.splitext(fpath[1])
	
	#in case of nifti was compressed as 'gz'
	if fname[1] == '.gz':
		fname = os.path.splitext(fname[0])

	if ext==None:
		new_fname = os.path.join(fpath[0], (fname[0] + suffix + fname[1]))
	else:
		new_fname = os.path.join(fpath[0], (fname[0] + suffix + ext))
	
	return os.path.normpath(new_fname)

def replaceDir(path, new_dir):
	# This function takes a file path and adds a suffix to the file name
	fpath = os.path.split(path)
	
	new_path = os.path.join(new_dir,fpath[1])
	
	return os.path.normpath(new_path)