# Exercise guidelines (8DC00)

The exercises and projects for this course are grouped by topic:

1. Image registration
2. Image segmentation
3. Computer aided design

Each topic's exercises are divided into several subsections, followed by the project.

The instructions for exercises and projects are all contained in Jupyter notebooks, which enables both text and code to be displayed (and executed) in a single document format.

Instructions are generally in the form of exercises, where you have to develop and test code, and questions, where you have to provide answers relevant to the code and tests you developed.

The code necessary to follow the instructions and to complete the exercises and projects is provided as a collection of Python files.


### Code and data repository structure

Once you have cloned this GitHub repository to your local machine, say to a folder named `8DC00`, you will find the following folder and file structure:

```bash
8DC00
.
|____code
| |____utilities.py
| |____registration.py
| |____registration_tests.py
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

`Code` is organised in Python modules per topic (e.g. `registration.py`), each containing the Python functions (either complete or to be completed by you) particular to the topic of the exercise. These modules are referred to from the relevant Jupyter notebooks, throughout the exercise steps.

You will often have to write tests to check the validity of your funtions. For this reason, a separate module (e.g. `registration_tests.py`) is available per exercise topic. These modules contain code, either complete or to be completed by you, for testing purposes.

The testing functions are often already called from within the Jupyter notebooks, although take note that these tests might fail if it does not yet contain the completed code. Helper functions are provided in the `utilities.py` module.

The Jupter `notebooks` contain all exercise and project instructions, mostly structured according to a narrative interspersed with code snippets and example figures. The [README](README.md) provides the order (with links) in which the exercises can be followed. 

Finally, the `data` folder contains all of the data necessary to complete the exercises and projects. Hardcoded filenames in the `_tests.py` modules are referenced to the `8DC00` folder. You might have to change these filenames if you program and run your code from outside the notebooks.


### Software installation

All program implementation has been done in Python. Python will be required to develop and test the code necessary for the exercises and projects.

The `requirements.txt` file provides a list of packages necessary to run the complete development environment for the exercises.

Once you have a Python environment set up, run the following from your terminal / command line

````bash
cd ./8DC00  # or your relevant local repository
pip install -r requirements.txt
````
in order to install the required packages.

### Development and testing

We recommend using Jupyter Notebook (recommended, covered in the [Essential Skills](https://github.com/tueimage/essential-skills) module) or [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) (a bit more advanced functionality that some might find it useful) to follow along the exercises and to run the example code. 

When programming functions and tests, you might want to use a Python development environment For this we suggest <>.

 
