from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='dyse',
    version='1.0',

    author='MeLoDy Lab',
    #author_email='',
    description='Dynamic System Explanation Framework',
    long_description='Tools for building, simulating, and verifying qualitative models, built by the Mechanisms and Logic of Dynamics Lab at the University of Pittsburgh',
    #license='',
    keywords='dynamic system boolean logical qualitative modeling simulation',

    packages=[
        'Sensitivity',
        'Simulation.Simulator_Python',
        'Translation',
        'Visualization'
    ],
    include_package_data=True,

    install_requires=[
        'matplotlib', # Visualization, Simulation
        'networkx', # Translation, Extension
        'numpy', # Simulation
        'openpyxl', # Extension, Filtration, Simulation, Visualization
        'pandas', # Classification, Extension, Simulation, Visualization
        'xlrd', # required by pandas
        'PyQt5==5.13', # GUI, QtWebEngine MacOS error with 5.14 and 5.15
        'seaborn', # Visualization
        'statsmodels',
        'sklearn'
    ],
    zip_safe=False # install as directory
    )
