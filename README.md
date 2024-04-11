Install depenencies needed to run code:

##Install GNU
File: https://www.gnu.org/software/gsl/
Instructions: https://gist.github.com/TysonRayJones/af7bedcdb8dc59868c7966232b4da903

##Install PyGSL
pygsl tar file: https://github.com/pygsl/pygsl/releases 
pip install pygsl or https://pypi.org/project/pygsl/
If want to set up manually (issues may occur):
cd desktop (or where your file is)
gzip -d -c pygsl-2.3.3.tar.gz | tar xvf -
cd pygsl-2.3.3
python setup.py gsl_wrappers
python setup.py config
sudo python setup.py install
