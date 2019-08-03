from setuptools import setup,find_packages

setup(
   name='ryd',
   version='0.1(dev)',
   description='Calculate properties of rydberg atoms and generate their hamiltonians in external fields',
   author='J.D.R Tommey',
   author_email='ucapdrt[at]ucl[dot]ac[dot]uk',
   url="https://github.com/jdrtommey/rydprop",
   packages=find_packages(),  #same as name
   install_requires=['numpy','scipy','numba','tqdm','pint'], #external packages as dependencies
)