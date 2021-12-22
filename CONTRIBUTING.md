# DySE Framework Contributing Guidelines

## Environment Setup

### Python
- USE VIRTUAL ENVIRONMENTS (Anaconda or venv)
- Install [Anaconda](https://www.anaconda.com/download/) 
    - Anaconda lets you set up multiple Python environments for testing dependencies, and can also install non-Python packages
    - (Windows) use `python -m pip install [package]` to avoid permission errors with pip
    - (Windows) add the Anaconda3 directory to the path to use conda commands in cmd.exe or Git Bash
        - Control Panel > System and Security > System (or Windows Key + Pause) > Advanced system settings > Environment Variables
        - Edit "Path"
        - add `;C:\Anaconda3;C:\Anaconda3\Scripts` to the existing path (use the path where you installed Anaconda3) 
- Create a Python 3.6 environment and activate the environment
    - from the shell
~~~shell
conda create -n py36 python=3.6
source activate py36
~~~
- Add the line `source activate py36` to your `~/.bash_profile` to always enable this environment in new shell windows/tabs
- Run `python --version` from the shell to check that the version is Python 3.6
- Note that you can export a virtual environment for someone else to use
    
### Git
- [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- See: [doc/Git_Tutorials_and_Resources.docx](doc/Git_Tutorials_and_Resources.docx)
- Tutorials:
    - [Atlassian Bitbucket Git Tutorial](https://www.atlassian.com/git/tutorials/learn-git-with-bitbucket-cloud) 
    - [Main git site tutorial](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control)
- From the shell: 
    - Create a folder called "framework" where you want to store the repository
    - Navigate to the directory containing "framework"
    - Clone the repository into the folder "framework" and checkout your branch (replace ${user} with your Bitbucket username and ${your-branch-name} with your branch name)
        - Make sure you are on your branch and not on `master` or `develop`
    - Occasionally pull in changes from `develop` to keep your branch up to date

~~~shell
git clone https://${user}@bitbucket.org/biodesignlab/framework.git framework
git checkout ${your-branch-name}
git status
~~~
- (Windows) [Git Bash](https://gitforwindows.org)
    - to change the start directory, right-click on git-bash.exe (or its shortcut) and change Properties > Start in: to the framework directory
    - to run python in git-bash, run `winpty python`
- (Linux) If getting the error message "package runit not configured yet":
~~~shell
sudo apt-get purge runit
sudo apt-get purge git-all
sudo apt-get purge git
sudo apt-get autoremove
sudo apt update
sudo apt install git
~~~

## Framework Setup

- See [doc/Git_Tutorials_and_Resources.docx](doc/Git_Tutorials_and_Resources.docx) for git setup and instructions on cloning the DySE framework repository from Bitbucket

- follow framework installation directions in [README.md](README.md)

- Optionally assign the framework path to a shell variable:
- Mac
~~~shell
export DYSEPATH="[absolute path to framework directory]" >> ~/.bash_profile
~~~

- Linux
~~~shell
export DYSEPATH="[absolute path to framework directory]" >> ~/.bashrc
~~~

- update the current shell 
~~~shell
source ~/.bash[_profile or rc]
~~~
- check the path
~~~shell
echo $DYSEPATH
~~~

## Code Development

- to import framework modules in Python after dyse installation:
~~~Python
from Simulation.Simulator_Python.simulator import Simulator
~~~

### Naming Conventions

- module_name
- function_name
- variable_name
- ClassName
- script-name

## Other Resources

### Jupyter Notebooks

- Installation:
~~~shell
pip install jupyter
~~~

- To use notebook extensions, like table of contents:
~~~shell
pip install jupyter-nbextensions-configurator
~~~

- If there are issues with your python version (using your virtual environment py36):
~~~shell
pip install --user ipykernel
python -m ipykernel install --user --name=py36
~~~
- Within the notebook, select Kernel > Change kernel > py36


- [http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb](http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb)
- [https://github.com/jupyter/jupyter/wiki/a-gallery-of-interesting-jupyter-notebooks](https://github.com/jupyter/jupyter/wiki/a-gallery-of-interesting-jupyter-notebooks)
- [http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html](http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html)

### Gitignore

- [https://www.gitignore.io](https://www.gitignore.io)

### Mac Python installation

- [https://www.davidculley.com/installing-python-on-a-mac/](https://www.davidculley.com/installing-python-on-a-mac/)
- (Mac) [Homebrew](https://brew.sh)

### More Git and Version Control

- Use `git stash` to temporarily store changes (e.g., you want to switch to a new branch or temporarily go back to the version at the last commit)
    - `git stash apply` to reapply changes
- `git checkout -- <file>` to revert a file to its version at the last commit
- [https://homes.cs.washington.edu/~mernst/advice/version-control.html](https://homes.cs.washington.edu/~mernst/advice/version-control.html)
- [https://git-scm.com/book/en/v2](https://git-scm.com/book/en/v2)
- [Gitflow]([https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow])

### Bitbucket Support

- [https://support.atlassian.com/bitbucket-cloud/](https://support.atlassian.com/bitbucket-cloud/)

### Python Style Guides

- PEP Style Guide
    - [https://www.python.org/dev/peps/pep-0008/](https://www.python.org/dev/peps/pep-0008/)
    - Use docstrings to describe functions
    - Comment code with high-level descriptions of what the code is doing and __WHY__
        - if the code itself is more readable, fewer comments are needed

- Google Python Style Guide
    - [https://google.github.io/styleguide/pyguide.html](https://google.github.io/styleguide/pyguide.html)

### Structuring Code

-	[http://docs.python-guide.org/en/latest/writing/structure/](http://docs.python-guide.org/en/latest/writing/structure/)
-	[http://python-packaging.readthedocs.io/en/latest/minimal.html](http://python-packaging.readthedocs.io/en/latest/minimal.html)
-	[http://intermediate-and-advanced-software-carpentry.readthedocs.io/en/latest/structuring-python.html](http://intermediate-and-advanced-software-carpentry.readthedocs.io/en/latest/structuring-python.html)
-	[https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/](https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/)

### Packaging

- [Python packaging tutorial](https://packaging.python.org/tutorials/distributing-packages/#packaging-your-project)
- use setuptools (see `setup.py`)
- add all non-Python files to `MANIFEST.in`
- to create a tarball distribution:
~~~shell
python setup.py sdist
~~~

- if making changes to existing .cpp packages (e.g., dishwrap), run `make` in the same directory as the .cpp and the makefile

### Code Performance

- Write for functionality first, then optimize
- [https://www.airpair.com/python/posts/top-mistakes-python-big-data-analytics](https://www.airpair.com/python/posts/top-mistakes-python-big-data-analytics)
- [https://wiki.python.org/moin/PythonSpeed/PerformanceTips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- Use NumPy and vectorization when working with arrays
- Cython for code optimization and C/C++ compatibility: [http://cython.org](http://cython.org)
- Performance of Cython and Numpy: [https://www.ibm.com/developerworks/community/blogs/jfp/entry/A_Comparison_Of_C_Julia_Python_Numba_Cython_Scipy_and_BLAS_on_LU_Factorization?lang=en](https://www.ibm.com/developerworks/community/blogs/jfp/entry/A_Comparison_Of_C_Julia_Python_Numba_Cython_Scipy_and_BLAS_on_LU_Factorization?lang=en)
- NumPy performance examples: [https://modelingguru.nasa.gov/docs/DOC-1762](https://modelingguru.nasa.gov/docs/DOC-1762)
- Coding language comparisons: 
    - Python, MATLAB, C, R, and others: [https://modelingguru.nasa.gov/docs/DOC-2625](https://modelingguru.nasa.gov/docs/DOC-2625)
    - Python, MATLAB, C, Java, and others: [https://modelingguru.nasa.gov/docs/DOC-2676](https://modelingguru.nasa.gov/docs/DOC-2676) 

### General Modeling/Coding

- [https://modelingguru.nasa.gov/index.jspa?view=overview](https://modelingguru.nasa.gov/index.jspa?view=overview)

### Code Licensing

- [https://choosealicense.com](https://choosealicense.com)

### GUI

- [http://www.tldp.org/LDP/LG/issue83/evans.html](http://www.tldp.org/LDP/LG/issue83/evans.html)
- create GUI layout in qt designer, save as .ui file
~~~shell
open -a Designer
~~~

- compile .ui file into a python class
~~~shell
pyuic5 -o simulator_gui_form.py simulator_gui.ui
~~~

- write GUI function code, importing the .py file for the ui
    - for reference, see code in Framework/GUI/simulator_gui.py
- after debugging, compile a standalone application using pyinstaller: [https://pyinstaller.readthedocs.io/en/stable/usage.html](https://pyinstaller.readthedocs.io/en/stable/usage.html)
~~~shell
pyinstaller --paths=${FRAMEWORKPATH} --onefile --windowed simulator_gui.py
~~~

- to use the existing spec file
~~~shell
pyinstaller simulator_gui.spec
~~~

- upgrade python packages causing problems during packaging `pip install --upgrade [package]`
    - especially setuptools
- if QtWebEngineWidgets is not found
    - may need to uninstall and reinstall pyqt5 and PyQtWebEngine
~~~shell
pip install PyQtWebEngine
~~~

- fixing problems with py2cytoscape and igraph
    - symptom message: "Library not loaded: @rpath/libxml2.2.dylib ... Reason: Incompatible library version: _igraph.cpython-36m-darwin.so requires version 12.0.0 or later, but libxml2.2.dylib provides version 10.0.0"
    - solution from https://chrisjcameron.github.io/2016/04/27/homebrew_and_anaconda/
~~~shell
brew unlink igraph
brew uninstall igraph
brew update
brew doctor
brew install igraph
pip install python-igraph --global-option=build_ext --global-option="-L/usr/lib:/usr/local/lib"
~~~

### Visualization

- Cytoscape for graph visualization
    - [Install Cytoscape](http://www.cytoscape.org) 
    - install py2cytoscape for access to the CyREST API 
        - [https://py2cytoscape.readthedocs.io/en/latest/](https://py2cytoscape.readthedocs.io/en/latest/)
        - [https://github.com/cytoscape/cytoscape-automation/tree/master/for-scripters/Python](https://github.com/cytoscape/cytoscape-automation/tree/master/for-scripters/Python)
    - Ono, Keiichiro, et al. "CyREST: Turbocharging Cytoscape Access for External Tools via a RESTful API." F1000Research 4 (2015).
    - [https://github.com/cytoscape/cytoscape-tutorials/wiki](https://github.com/cytoscape/cytoscape-tutorials/wiki)
- [http://www.vischeck.com/vischeck/](http://www.vischeck.com/vischeck/)

### Other tools

- INDRA : follow instructions on [https://github.com/sorgerlab/indra](https://github.com/sorgerlab/indra)
    - With EIDOS : [https://gist.github.com/bgyori/37c55681bd1a6e1a2fb6634faf255d60](https://gist.github.com/bgyori/37c55681bd1a6e1a2fb6634faf255d60)
- [https://github.com/clulab](https://github.com/clulab)
- EIDOS : [https://github.com/clulab/eidos](https://github.com/clulab/eidos)

___
# README template

## [Name of package]

### Description of Files

- `file.py` 
    - what does this module do 

### Usage
- copy usage statement below
~~~shell
Usage:
~~~

### Examples
- show how to call modules with example files, and/or point to example scripts in `examples/`
~~~shell
python file.py example_input
~~~