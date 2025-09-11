# Installing the project from scratch

## ***Requires python 3.11 (not more, for compatibility with open3d package)***

### <u>For an offline installation on your own PC</u>

Choose or create a directory where you will install the requirements stuff

In the rest of the installation, we'll call it **WORKING_DIR**

- Download the project from the Jupyterhub ("03_stereo_imaging") or Github (stereo-training-course) repo into your WORKING_DIR directory.

This is what we will name your **WORKSPACE** directory

ex :
From the Github repo
> WORKSPACE = stereo-training-course 

From the Jupyterhub repo
> WORKSPACE = 03_stereo_imaging

***/!\ Important : for the following steps, when talking about a cmd shell and if you're on Windows, 
use the Windows Terminal console ('cmd') and NOT the Powershell console***

- Install Jupyter in in your global python environment : from a cmd shell, run :

> pip install notebook

- Create a folder "stereo-kernel" on your computer, in your WORKING_DIR directory, in which the kernel will be installed 

After having creating it, you might have a folder here :
> WORKING_DIR\stereo-kernel

- Go to this folder, and from a cmd shell, create a new virtual environement :

> WORKING_DIR\stereo-kernel>python -m venv stereo-venv

- Then activate it :

<u>On Windows :</u>
WORKING_DIR\stereo-kernel>stereo-venv\Scripts\activate

<u>On Linux :</u>
WORKING_DIR\stereo-kernel>source stereo-venv\bin\activate

- Here is the state in where you should be :

> (stereo-venv) WORKING_DIR\stereo-kernel>

- In this new virtual environment, install ipykernel :

> (stereo-venv) WORKING_DIR\stereo-kernel>pip install ipykernel

Then link it to the jupyter kernel :

> (stereo-venv) WORKING_DIR\stereo-kernel>python -m ipykernel install --user --name stereo-venv

<u>You should get this kind of message :</u>
***Installed kernelspec stereo-venv in WORKING_DIR\AppData\Roaming\jupyter\kernels\stereo-venv***

- Install packages required :

> pip install numpy
> pip install opencv-python
> pip install open3d
> pip install laspy[lazrs,laszip]
> pip install matplotlib
> pip install scikit-image
> pip install scipy

- Go to your workspace folder (the one in which you have   'WORKING_DIR\dev\stereo-notebook'), and from a cmd shell, launch jupyter :

> WORKING_DIR\WORKSPACE>jupyter notebook

- Select as associated kernel the "stereo-venv" just created

You may be able to run on your own PC !

