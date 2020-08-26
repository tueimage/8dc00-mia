# Setting up the Python environment

To get started with setting up a Python environment, follow the instructions in the Getting Started section of the [Essential Skills](https://github.com/tueimage/essential-skills/blob/master/python-essentials.md) Python module. The Anaconda distribution is recommended. Optionally, you can use the desktop GUI called Anaconda Navigator.

To run the complete development environment for this course, you need to install five additional Python packages: `matplotlib`, `jupyter`, `scikit-learn`, `scipy` and `spyder`. It is recommended to install these packages in a separate Conda environment. A Conda environment is a directory in which you can install files and packages such that their dependencies will not interact with other environments, which is very useful if you develop code for different courses or research projects. If you want to know more about how to use Conda, read the [docs](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

After installing Anaconda, you can create a new Conda environment and install the required packages by running the following commands from the Anaconda Prompt application:

````bash
conda create --name 8dc00 python=3.6				# create a new environment called `8dc00`
conda activate 8dc00						# activate this environment 
conda install matplotlib jupyter scikit-learn scipy spyder	# install the required packages
````

NB: You have to activate the `8dc00` environment every time you start working on the assignments (`conda activate 8dc00`).


## Getting started with the exercises and projects

We recommend using *Jupyter Notebook* to follow the exercises and run the example code (also see the [Essential Skills](https://github.com/tueimage/essential-skills) module). An alternative is [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) which has a bit more advanced functionality that some might find useful.

In order to start Jupyter Notebook, type `jupyter notebook` in Anadonda Prompt (after activating the `8dc00` environment with `conda activate 8dc00`). It is best if you change the directory to the directory containing the code before starting Jupyter Notebook. Similarly, you can start the integrated development environment *Spyder* by typing `spyder` in the Anaconda Prompt.

Tip: In Jupyter Notebooks, modules are not automatically re-imported after you changed something in a function, so make sure to also run the first cell in each Notebook that contains the `%autoreload` command (or simply restart the kernel).


## Code and data repository structure

To get started, you have to clone this GitHub repository to your local machine, say to a folder named `8DC00`. Here you will find the following folder and file structure:

```bash
8DC00
.
|____code
| |____registration.py
| |____registration_tests.py
| |____registration_util.py
| |____registration_project.py
| |____...
|____data
|____notebooks
| |____registration_introduction.ipynb
| |____registration_exercise_(x).ipynb
| |____registration_project.ipynb
| |____...
|____guidelines.md
|____README.md
|____rubric.md
|____requirements.txt
```

`Code` is organised in Python modules per topic (e.g. `registration.py`), each containing the Python functions (either complete or to be completed by you) particular to the topic of the exercise. These modules are referred to from the relevant Jupyter notebooks.

The testing functions (e.g. `registration_tests.py`) can be used validate the code that you developed. These functions are often already called from within the Jupyter notebooks, although some of these tests might fail if they do not yet contain completed code. Helper functions are provided in the `utilities.py` module.

The Jupter `notebooks` contain all exercise and project instructions, mostly structured according to a narrative interspersed with code snippets and example figures. The [README](README.md) provides the order (with links) in which the exercises can be followed. 

Finally, the `data` folder contains all of the data necessary to complete the exercises and projects. Hardcoded filenames in the `_tests.py` modules are referenced to the `8DC00` folder. You might have to change these filenames if you program and run your code from outside the notebooks.


 
