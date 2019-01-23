# DRIR Learning
A python console application that enables machine learning on 
directional room impulse responses from microphone array recordings or simulations.  

## 1 Installation
It is recommended to create and activate a new and empty conda environment:
```
conda create --name myenv python=3.6
source activate myenv # linux, mac
activate myenv # windows
```
Clone the project:
```
git clone https://github.com/Agent49/drirlearning.git
```
Install the requirements:
```
pip install -r requirements.tx
```
*Tensorflow* requires python 3.4 to 3.6. It is highly recommended to run the application with GPU-support. 
For more information see:

https://www.tensorflow.org/install/gpu

## 2 Run application from console
You can run the application from console. 
Optional parameters are useful if you want to quickly switch
from small to huge data sets or change some hyperparameters.
For more information type:
```
python ./drirlearning.py -h
```

## 3 Assess and compare models
The best way to assess and compare the performance of your models is a visual inspection
of loss and other results with TensorBoard. Therefore, the last line on your console gives you
instructions after the process has finished. 

## 4 Write and run your own models Write and run your own models
The purpose of this application is that you create your own models/neural nets
and train them on different data sets. Initial models and sample data is provided.
The principle steps are as follows:

1. Create your own model in *model.py*.
2. In *drirlearning.run()* call function *utils.run_model(your_model...)*,
given your models as a callback function.  
3. Adjust the configuration hard-coded or via CLI interface.
4. Run the application.

The modul *drirlearning.utils.py* will provide you with helper functions.
You can build the complete documentation for your browser with search functions
by typing:
 ```
cd ./docs
make html
```
