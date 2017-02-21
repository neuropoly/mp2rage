# mp2rage

Scripts to compute T1 maps from MP2RAGE images.

-------------------------
Installation instructions
-------------------------

Beforehand, you need to install:

    - the SpinalCordToolbox (https://sourceforge.net/p/spinalcordtoolbox/wiki/install_from_package/)
	- the library PyMRT (https://pypi.python.org/pypi/pymrt) with the SpinalCordToolbox python (spinalcordtoolbox/python/bin/python) running in a terminal window the following command (you might need the administrator privileges, for this, just add "sudo" at the beginning of the following command):
    pip install pymrt


Then, you need to replace the 5 following scripts in your PyMRT installation folder by the scripts given in this repository:
	utils.py 
	computation.py           
	__init__.py
	naming.py
	sequences/mp2rage.py
To do so:
    - go into the PyMRT installation folder: spinalcordtoolbox/python/lib/python2.7/site-packages/pymrt/
    - copy the following scripts from this Github repository:
	
	utils.py                     AND PASTE IT IN         ./
	computation.py               AND PASTE IT IN         ./
	__init__.py                  AND PASTE IT IN         ./
	naming.py                    AND PASTE IT IN         ./
	sequences/mp2rage.py         AND PASTE IT IN         ./sequence/
	
The main script to compute T1 maps from the UNI MP2RAGE image is named mp2rage_compute_t1_map.py. If you want to be able to run it from any folder (in your Terminal window), please add the folder including this script ("mp2rage" git directory) to your .bash_profile. To do so:
    - open a Terminal window
	- to go to your home directory, type: cd
	- to edit your .bash_profile, type: vi .bash_profile
	- press key "i"
	- add the following line: PATH=${PATH}: <path_to_your_MP2RAGE_git_directory>/mp2rage
	- press key "ESC"
	- to save and quit, press keys: ":", "w", "q", "ENTER". 
	- open a new Terminal window to load your new .bash_profile


-----
Usage
-----

To see help, open a Terminal window and type:
mp2rage_compute_t1_map.py -h

Here is an example of a command to run the script and compute the T1 map from the UNI MP2RAGE image:
mp2rage_compute_t1_map.py -i uni.nii.gz -acqparam ti=700,1500:alpha=7,5:tr_seq=4000:tr_gre=2.84:matrix_sizes=184,184,160:bandwidths=750,750:part_fourier_factors=1.0,6/8,1.0:grappa_refs=0,32,0:grappa_factors=1,2,1 -t1range 100,3000 -o T1map.nii.gz,RHOmap.nii.gz


