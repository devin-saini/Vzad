# Work in Progress
Install depenencies needed to run code:

## Install NLOPT (if nlopt can't be downloaded from respective IDE)
Install Swig (dependency for numpy)
brew install swig

Download nlopt zip file, nlopt homepage: https://nlopt.readthedocs.io/en/latest/
After downloading and unzipping the nlopt zip file, from the command line move into the nlopt folder and enter the following promts:
mkdir build
cd build

Move the cmake.txt file into the "build" folder

https://stackoverflow.com/questions/62704802/cannot-install-nlopt-python-module

# DONT FOLLOW STEPS BELOW (These are for c++ vzad)
Install GNU
File: https://www.gnu.org/software/gsl/
Instructions: https://gist.github.com/TysonRayJones/af7bedcdb8dc59868c7966232b4da903

Install PyGSL
pygsl tar file: https://github.com/pygsl/pygsl/releases 
pip install pygsl or https://pypi.org/project/pygsl/
If want to set up manually (issues may occur):
cd desktop (or where your file is)
gzip -d -c pygsl-2.3.3.tar.gz | tar xvf -
cd pygsl-2.3.3
python setup.py gsl_wrappers
python setup.py config
sudo python setup.py install
