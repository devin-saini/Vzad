# Work in Progress
The PyCharm IDE is recommended as it can be used to import all dependencies needed to run code (https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html)
To manually install depenencies needed to run code, please refer to the instructions below.

The dependencies needed are:
OS to interact with the operating system, 
Numpy for data analysis, 
Matplotlib for graphing, 
Scipy for fourier transform, 
nlopt for analysis algorithms, 
Glob2 for paths of specific files

## Install NLOPT (if nlopt can't be downloaded from respective IDE)
Install Swig (dependency for nlopt) by going to swig hompeage (https://www.swig.org/download.html) or using homebrew
Install homebrew by following directions from https://brew.sh/ 
brew install swig

Intall numpy (dependency needed for nlopt), numpy homepage: https://numpy.org/

Download nlopt zip file, nlopt homepage: https://nlopt.readthedocs.io/en/latest/
After downloading and unzipping the nlopt zip file, from the command line move into the nlopt folder and enter the following promts:
cmake -DNLOPT_GUILE=OFF -DNLOPT_MATLAB=OFF -DNLOPT_OCTAVE=OFF -DNLOPT_TESTS=OFF -DPYTHON_EXECUTABLE=/usr/local/bin/python3 or cmake -DPython_EXECUTABLE=/usr/bin/python3.12
make
sudo make install

See if python bindings can me imported
python3 -c "import nlopt; print(nlopt.__version__)"


https://stackoverflow.com/questions/62704802/cannot-install-nlopt-python-module
