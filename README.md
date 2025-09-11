# Installing the project from scratch

## ***Requires python 3.11 (not more, for compatibility with open3d package)***

### <u>For an offline installation on your own PC</u>

Choose or create a 'root' directory where you will install the requirements stuff

For instance,
    - if on Windows : it might be located here : HOME = "C:\Users\buddy" 
    
    - if on Linux, HOME : "/home/buddy"

- Download the project from the Jupyterhub ("03_stereo_imaging") or Github (stereo-training-course) repo
into your root directory.
<u>This is what we will name your workspace  directory</u>

ex : 
> HOME\dev\stereo-notebook


***/!\ Important : for the following steps, when talking about a cmd shell and if you're on Windows, 
use the Windows Terminal console ('cmd') and NOT the Powershell console***

- Install Jupyter in in your global python environment : from a cmd shell, run :

> pip install notebook

- Create or select a folder "stereo-kernel" on your computer, whereever you want, in which the kernel will be installed :

For instance,
    - if on Windows : it might be located here : HOME = "C:\Users\buddy" 
    
    - if on Linux, HOME : "/home/buddy"

After having creating it, you might have a folder here :
HOME\stereo-kernel

- Go to this folder, and from a cmd shell, create a new virtual environement :

> HOME\stereo-kernel>python -m venv stereo-venv

- Then activate it :

<u>On Windows :</u>
HOME\stereo-kernel>stereo-venv\Scripts\activate

<u>On Linux :</u>
HOME\stereo-kernel>source stereo-venv\bin\activate

- Here is the state in where you should be :

> (stereo-venv) HOME\stereo-kernel>

- In this new virtual environment, install ipykernel :

> (stereo-venv) HOME\stereo-kernel>pip install ipykernel

Then link it to the jupyter kernel :

> (stereo-venv) HOME\stereo-kernel>python -m ipykernel install --user --name stereo-venv

<u>You should get this kind of message :</u>
***Installed kernelspec stereo-venv in HOME\AppData\Roaming\jupyter\kernels\stereo-venv***

- Install packages required :

> pip install numpy
> pip install opencv-python
> pip install open3d
> pip install laspy[lazrs,laszip]
> pip install matplotlib
> pip install scikit-image
> pip install scipy

- Go to your workspace folder (say 'HOME\dev\stereo-notebook'), and from a cmd shell, launch jupyter :

> HOME\dev\stereo-notebook>jupyter notebook

- Select as associated kernel the "stereo-venv" just created

You may be able to run on your own PC !

