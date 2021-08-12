# Amharic Speech-to-Text engine
## Introduction
<p> Speech recognition technology allows for hands-free control of smartphones, speakers, and even vehicles in a wide variety of languages. Companies have moved towards the goal of enabling machines to understand and respond to more and more of our verbalized commands. There are many matured speech recognition systems available, such as Google Assistant, Amazon Alexa, and Apple’s Siri. However, all of those voice assistants work for limited languages only. </p>

<p>The World Food Program wants to deploy an intelligent form that collects nutritional information of food bought and sold at markets in two different countries in Africa - Ethiopia and Kenya. The design of this intelligent form requires selected people to install an app on their mobile phone, and whenever they buy food, they use their voice to activate the app to register the list of items they just bought in their own language. The intelligent systems in the app are expected to live to transcribe the speech-to-text and organize the information in an easy-to-process way in a database. </p>

<p>Our responsibility was to build a deep learning model that is capable of transcribing a speech to text in the Amharic language. The model we produce will be accurate and is robust against background noise.</p>
 
<h1> Key Topics</h1>
the following  topics will be covered in this unit:
<b>1.Data pre-processing </b>
 .Load audio file
 .Load transcriptions
 .Convert into channels 
 .Standardize sampling rate
 .Resize to the same length
 .Data argumentation
 .Feature extraction: 
 .Acoustic modeling:


<b>2.Modelling and Deployment using MLOps </b>
.Modeling: Build a Deep learning model that converts speech to text.
.Choose one of deep learning architecture for speech recognition
    Use Connectionist Temporal Classification Algorithm for training and inference 
    CTC takes the character probabilities output of the last hidden layer and derives the correct sequence of characters
.Evaluate your model. 
.Effect of data augmentation: apply different data augmentation techniques and version all of them in DVC. Train model for using these data and study the effect of data    augmentation on the generalization of the model.
.Model space exploration: using hyperparameter optimization and by slightly modifying the architecture e.g. increasing and decreasing the number of layers to find the best model. 
.Write test units that can be run with CML that will help code reviewers accept Pull Requests (PRs) based on performance gain and other crucial elements. 
.Version different models and track performance through MLFlow
.Evaluate the model using evaluation metrics for speech recognition Word error rate (WER)

<b>3.Serving predictions on a web interface</b>

<h1>Learning Objectives</h1>
<b>Skills:</b>
Working with audio as well as text files
Familiarity with the deep learning architecture
Model management (building ML catalog containing models, feature labels, and training model version)
MLOps  with DVC, CML, and MLFlow

<b>Knowledge:</b>
Audio and text processing 
Deep learning methods (TensorFlow, Keras, Pytorch ) 
Hyperparameter tuning
Model comparison & selection
Experiment Analysis




Structure
├── logs
├── modules
├── notebooks
├── tests
└── Dockerfile

<h1>Helpful Links</h1>
https://arxiv.org/pdf/2103.07762.pdf
https://www.tutorialspoint.com/digital_signal_processing/index.htm
https://librosa.org/doc/main/feature.html
https://heartbeat.fritz.ai/the-3-deep-learning-frameworks-for-end-to-end-speech-recognition-that-power-your-devices-37b891ddc380
https://heartbeat.fritz.ai/the-3-deep-learning-frameworks-for-end-to-end-speech-recognition-that-power-your-devices-37b891ddc380

# Contributors

* [Azaria Tamrat](https://github.com/Azariagmt)
* [Bethelhem Sisay](https://github.com/Bethelsis)
* [Daniel Zelalem](https://github.com/daniEL2371)
* [Dorothy Cheruiyot](https://github.com/Doro97)
* [Natneal Teshome](https://github.com/Natty-star)
* [UWASE Rachel](https://github.com/ntabanarachel)
* [Yosef Alemneh](https://github.com/mozartofmath)



