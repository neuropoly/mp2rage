# T1 mapping with MP2RAGE sequence

Scripts to compute T1 maps from MP2RAGE data. This code is based on the excellent [PyMRT library](https://pypi.org/project/pymrt/). Note that the output T1 map is slightly different than that from the Siemens MP2RAGE sequence due to the different way of processing the data. More extensive comparison is ongoing. Please contact the authors for more info.

## Installation instructions

These scripts work with Python 2.7.

**STEP 1** : Install anaconda environment: https://www.continuum.io/downloads

**STEP 2** : Install pymrt:
~~~
pip install pymrt==0.0.1.3
~~~

**STEP 3** : 
Replace the 5 following scripts in your PyMRT installation folder by the scripts given in this repository: **utils.py**, **computation.py**, **__init__.py**, **naming.py** and **sequences/mp2rage.py**. To do so:

    - go to the folder where PyMRT package was installed (to find the path to this folder, type in a Terminal window: python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
    - copy the following scripts from this Github repository:
	
	utils.py                     AND PASTE IT IN         ./pymrt/
	computation.py               AND PASTE IT IN         ./pymrt/
	__init__.py                  AND PASTE IT IN         ./pymrt/
	naming.py                    AND PASTE IT IN         ./pymrt/
	sequences/mp2rage.py         AND PASTE IT IN         ./pymrt/sequence/

**STEP 4** :
The main script to compute T1 maps from the UNI MP2RAGE image is named `mp2rage_compute_t1_map.py`. To be able to run it from any folder (in your Terminal window), please add the folder including this script (`mp2rage` git directory) to your `.bash_profile`. To do so:

    - open a Terminal window
    - to go to your home directory, type: cd
    - to edit your .bash_profile, type: vi .bash_profile
    - press key "i"
    - add the following line: PATH=${PATH}: <path_to_your_MP2RAGE_git_directory>/mp2rage
    - press key "ESC"
    - to save and quit, press keys: ":", "w", "q", "ENTER". 
    - open a new Terminal window to load your new .bash_profile

**NOTE :** 
Make sure you always have both mp2rage_compute_t1_map.py & parser.py in the same folder.

## Usage

To see help, open a Terminal window and type:
~~~
mp2rage_compute_t1_map.py -h
~~~

Here is an example of a command to run the script and compute the T1 map from the UNI MP2RAGE image:
~~~
mp2rage_compute_t1_map.py -i uni.nii.gz -acqparam ti=700,1500:alpha=7,5:tr_seq=4000:tr_gre=2.84:matrix_sizes=184,184,160:bandwidths=750,750:part_fourier_factors=1.0,6/8,1.0:grappa_refs=0,32,0:grappa_factors=1,2,1 -t1range 100,3000 -o T1map.nii.gz,RHOmap.nii.gz
~~~

If you experience any issue or would like additional help, please post an issue [there](https://github.com/neuropoly/mp2rage/issues).

## Authors

Simon Lévy, Riccardo Metere, Julien Cohen-Adad

## Copyright

The MIT License (MIT)

Copyright (c) 2018 École Polytechnique, Université de Montréal

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
