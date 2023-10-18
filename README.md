# Adversarial attack generator for general accessories and real-time testing GUI

<!-- ![A picture of a man's face](/readme_assets/Ivan.PNG) ![The same man, but with a patterned mask on his face](/readme_assets/Ivan19thperturb.PNG) -->

<p align="center">
  <img src="/readme_assets/Ivan.PNG" />
  <img src="/readme_assets/Ivan19thperturb.PNG" />
</p>

Our capstone project trains an adversarial pattern within a constrained accessory area to dodge a person's true class or impersonate another class. From this repository, you can train an adversarial accessory against deepface, a state of the art facial recognition system, and test it out physically through the interactive GUI. 

The classifications supported for dodging and impersonation are:
* Gender (male, female)
* Ethnicity (white, asian, black, middle eastern, latino hispanic, indian)
* Mood (neutral, happy, sad, disgust, surprise, angry, fear)

[Project management](https://team31-399.atlassian.net/jira/software/projects/XW399/boards/1/timeline?selectedIssue=XW399-86&shared=&atlOrigin=eyJpIjoiYWM4OGJiNjc2YmRiNGNiMGIxNjI5ZGEzMWYyNzQ1YmEiLCJwIjoiaiJ9)

## Set up

### Environment and packages

We recommend the use of Python 3.8.10, which can be downloaded [here](https://www.python.org/downloads/release/python-3810/)

After cloning this repository, create a virtual environment

Conda (recommended, requires installation of conda)
```
conda create --name adversarial_env python=3.8
conda activate .adversarial_env
```
Venv (no installation required) -- Windows (powershell)
```
python3.8 -m venv .adversarial_env
\path\to\venv\Scripts\Activate.ps1 
```
Venv (no installation required) -- Mac (bash)
```
python3.8 -m venv .adversarial_env
source /path/to/venv/bin/activate
```

Install requirements:
```
pip install -r requirements.txt
```

### GPU Support

You can run this project without GPU support, but it will defintley be faster if you use your GPU.
For the tensorflow version used in this project, cuDNN=8.1 and CUDA=11.2 are needed. 
Installation guides:
* [Manual installation](https://spltech.co.uk/how-to-install-tensorflow-2-5-with-cuda-11-2-and-cudnn-8-1-for-windows-10/)
* [Installation with Docker](https://spltech.co.uk/how-to-setup-tensorflow-on-ubuntu-linux-with-multiple-gpus-using-docker/)

## Getting started

Try running the following command to create a simple adversarial facemask which impersonates female:
<br/>
If using database to store images:
```
python experiment\run_experiment.py -a facetest -c gender -d="Faces.db" -n 12 -i 20 -l 0 -p 0 -g 0 -s 50 -r .001 -H 15 -V 15 -R 10 -md impersonation -t female
```

If using folder with json file to store images:
```
python experiment\run_experiment.py -a facetest -c gender -d="Images" -j="Faces.json" -n 12 -i 20 -l 0 -p 0 -g 0 -s 50 -r .001 -H 15 -V 15 -R 10 -md impersonation -t female
```
<br/>
In your terminal, you should see some feedback as the program parses the image files and trains the perturbation over 20 iterations. The output adversarial pattern and graph of confidence in target class over iterations should pop up at the program end.

![A man with a facemask on his face. The facemask pattern is unpredictable and doesn't resemble any real object](/readme_assets/Ivan19thperturb.PNG) ![A graph that shows confidence in target class increasing over iterations. The bottom test reads "Classified: Woman, Confidence: 0.999](/readme_assets/confoverepoch.png)

If this runs smoothly without bugs, then you are ready to start experimenting!

## Modifying parameters

Below is the list of hyperparameters that can be modified for you experimentation and specific use cases.

### Mode: "-md" "--mode"

Whether your attack is to dodge the image subject's true class, or impersonate another class. Note: if dodging, input images should be subsetted to a single class.
Value: Either "dodge" or "impersonation"

### Classification: "-c" "--classification"

What classification you want to attack
Value: "gender", "ethnicity", or "emotion"

### Target: "-t", "--target"

If the mode is impersonation, this defines the target class that the adversarial attack will be converging to. 
Value:  
* for gender: male, female
* for ethnicity: white, asian, black, middle eastern, latino hispanic, indian
* for mood: neutral, happy, sad, disgust, surprise, angry, fear


### Accessory Type: "-a" "--accessory_type"

What accessory shape the adversarial attack should take.
Value: use "facemask", "glasses", "bandana" or "earrings" to use the accessories he have provided. Alternatively, you can create your own accessory by drawing a black design on a white background, saving it to the experiment/assets folder, and passing the filename.

### Image Directory: "-d" "--images_dir"

Where the adversarial training images are stored. 
Value: a valid file path, either to a .db SQL file or a directory. If you are using a directory, a json directory must be specified. 

### JSON Directory: "-j" "--json_dir"

Where true class labels for images are stored, if using images from a directory, not a database.
Value: a valid file path to a json file. See 'Faces.json' for an example of how image labels should be organised.

### Number of Images: "-n", "--num_images"

The number of images you want to use to train your adversarial attack. If this is less than the total number of images in the database or directory, then a random sample of this size is taken. Note: the actual number of images may decrease after preprocessing if opencv cannot find a face in the image given.
Value: a positive integer less than or equal to the total number of images in your given database or directory.

### Maximum Iterations: "-i", "--max_iter"

The maximum number of iterations of adversarial training. Increasing this may help your adversarial pattern achieve higher confidence in target class/lower confidence in true class, but if the attack gets stuck in a local optimum then you will need to wait longer or terminate without seeing the pattern. 
Value: a positive integer.

### Stop Probability: "-P", "--stop_probability"

If the confidence of the true class passes below (dodging) or if confidence in target class passes above 1-P (impersonation), the adversarial attack generator will stop and output the result (before max iterations reached)
Value: a positive real number, between 0 and 1.

### Step Size: "-s", "--step_size"

The learning rate used in gradient descent. A larger step size can make the adversarial attack in fewer iterations, with a greater variation of pattern, but can get stuck osciliating back and forth in some cases. Note: this hyperparameter is normalised by the number of images, so it must always be larger than the numebr of images.
Value: a real number.

### Decay Rate: "-r", "--decay_rate"

Reduces step size exponentially proportional to the number of iterations done. This means larger learning steps can be taken at the beginning of the gradient descent process, and smaller steps at the end. 
Value: a positive real number, usually in the range 0.00001 - 0.001.

### Total Variation Lambda: "-l", "--lambda_tv"

The weight of the total variation gradient in perturbation calculation. A higher lambda restricts how varied the colour values of the adversarial attack can be. Note: this hyperparameter is normalised by the number of images, so it must always be larger than the numebr of images.
Value: a real number.

### Printability Coefficient: "-p", "--printability_coeff"

The weight of the non-printability gradient in perturbation calculation. A higher printability coefficient means the possible colour values of the adversarial attack are more restricted by their ability to be printed. 
Value: a real number.

### Momentum Coefficient: "-m", "--momentum_coeff"

Speeds up learning in directions of low loss curvature without becoming unstable in directions of high loss curvature. Overall, can help generate the adversarial attack in fewer iterations.
Value: A real number, typically small.

### Gaussian Filtering: "-g", "--gauss_filtering"

Smooths out adversarial attack pattern to be more eye-pleasing and less obviously computer-generated.
Value: a real number between 0 and 1

### Brightness and Constrast Variation: "-b", "--bright_con_variation"

Adjusts the brightness and contrast of training images per iteration to increase robustness to differences in camera and printing.
Value: a positive real number, typically between 0 and 20

### Horizontal Movement: "-H", "--horizontal_move"

Adjusts accessory placement on face horizontally by a random number of pixels within the range of 0 and the paramenter, per iteration to make the adversarial attack more robust to differences in placement in real life.
Value: a positive integer, typically between 0 and 50.

### Vertical Movement: "-V", "--vertical_move"

Adjusts accessory placement on face vertically by a random number of pixels within the range of 0 and the paramenter, per iteration to make the adversarial attack more robust to differences in placement in real life.
Value: a positive integer, typically between 0 and 50.

### Rotational Movement: "-R", "--rotational_move"

Rotates accessory placement on face by a random degree within the range of 0 and the paramenter, per iteration to make the adversarial attack more robust to differences in placement in real life.
Value: a positive integer, typically between 0 and 40.

### Verbosity: "-v", "--v"

Controls whether interim output is displayed to the user. Default is True.
Value: "True" or "False".

## Interactive GUI


## References
Base of this project's code was sourced from Sharif et al. and translated to Python:

@inproceedings{Sharif16AdvML,
  author =       {Mahmood Sharif and Sruti Bhagavatula and Lujo Bauer 
						and Michael K. Reiter},
  title =        {Accessorize to a crime: {R}eal and stealthy attacks 
  						on state-of-the-art face recognition},
  booktitle =    {Proceedings of the 23rd ACM SIGSAC Conference on 
						Computer and Communications Security},
  year =         2016
} 
[Github](https://github.com/mahmoods01/accessorize-to-a-crime)
