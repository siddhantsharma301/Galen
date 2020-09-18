# Galen: A New Way to Diagnose Medical Images 
Siddhant Sharma, 2019

## Why
Hospitals are the go-to place for any medical issues or emergencies. However,
medical misdiagnosis and other general mistakes make up a large portion of medical
deaths in the United States. Medical errors are the third most common cause of medical
deaths in the US. This equates to almost 250,000 deaths per year caused by preventable
circumstances. Human doctors often make mistakes, especially if rare or new conditions
appear. In certain conditions, time is extremely sensitive in the recovery of the patient.
Doctors often spend crucial time examining and diagnosing medical reports that do not
contain issues instead of spending time on more crucial cases. Although computer models
can be used to determine whether reports contain issues, these traditional models often
have the problem of only being able to identify the “top X” number of conditions. To
effectively detect any potential issues in medical images and save doctors’ precious time, a
different type of computer detection system should be implemented. Instead of focusing on
a pathology-by-pathology basis, a focus should be placed on detecting general anomalies in
images. By placing an emphasis on anomaly detection, images without diseases can be
filtered out and images with diseases can be furthered investigated.

## How?
An example of an anomaly detection network is a variational autoencoder (VAE). This form of machine learning uses an extremely simple idea to find anomalies: find the differences. A VAE takes an input image and tries to reconstruct it, giving an output image. The goal of the VAE is to make the output image similar to the input image. By using this simple characteristic, a VAE can be trained on “normal” images to find anomalies. If a VAE is trained on a dataset of healthy chest x-rays, when given an unhealthy chest x-ray, it should fail to create a proper output. By using image similarity algorithms, this image similarity can be measured and used to determine whether an input-output pair is healthy or anomalous. I propose the use of a variational autoencoder to use anomaly detection, not classification, for healthcare.

## Training GIF
![Galen Training Demo](assets/training.gif)  

## How to Run the Model?
First, create a virtual environment in the parent directory of the actual code (AKA the directory that this readme is in). Example: `python3 -m venv galen`. Upon creating the virtual environment, activate it using `source galen/bin/activate` OUTSIDE of this parent directory. Then, `cd` back into this directory and run the following command: `pip3 -e .`. This adds the `galen` module to allow imports. Then, install dependencies and you should be good to go!