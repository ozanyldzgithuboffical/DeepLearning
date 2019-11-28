# DeepLearning
This repo contains the basis of the Deep Learning and Artificial Neural Network (ANN)

## Contents
  1. **Deep Learning & Artificial Neural Network (ANN)**
  - **Neuron Science & Computer Science**
  - **Neuron**
  - **Synapsis**
  - **ANN Logic**
  - **Graident Descent**
  - **Sthocastic Gradient Descent**
  - **Backpropogation**
  - **Forwardpropogation**
  
## Neuron Science & Role of Neurons & Synapsis in Intelligent Systems
- In human body neurons affect the decision taking,forming the actions even the reflexes.From this angle,neurans are included in terms of human thinking and acting.
- So,if neurons have such functions,then it asked that whether we can apply neuron modeling into machines to provide to create human thinking machines.
- Our nerve cells are consist of neuron and synapsis.When the signal traffic is passed through the neurons a new action is formed.Then another one is taken rapidly.
- There are input signals entering the nerve cells.Let call per signal as **x of N signal samples**.x of N elements input is taken as **signal**.These signals are used in a blackbox space from a function to obtain an output.
- **Synapsis** are used to carry the signals to neurons with a weight.So,every signal is powered by a weight.Let's call per weight is **w of M weight samples**.So,total signal is composed together with this formula: **w1x1+w2x2+...+wmxn , where n£N,m£M**
- Like other machine learning algorithms require,in neural network data must be standardized.Formally,each signal input should take a value in range 0-1,but it can be binomial or categorical also.
- Plus,there can be multi-variant systems which means that there can be multiple output.
- To obtain the output,we need a function.Let's call it **f(xi),where x £ N**.This function calculates the output.
- To activate the neuron we need an **activation function**.There many types of activation functions.Let's call it F and it takes weighted input signal as parameter.So, **F[(w1x1+w2x2+...+wmxn)]**

## Activation Functions
- To activate the signal to produce an output we need an **activation function**
- Activation function can be any function like sigmoid,sinus,cosinus,x2,x3 etc.However there are couple of mostly used activation functions which provide an advantage to ANNs.
- Some of them is listed here:
- **1.Step Function**
- Step function changes from 0 to 1 when a determined threshold value is exceeded.
- **2.Sigmoid Function**
- Sigmoid function changes from 0 , 0<F(x)<1 to 1.The formula of function is **F(X)=1/1+pow(e,-x)**
- **3.Rectifier Function**
-Rectifier function changes from 0, 0<F(x)<1 to max(x,0)
- **4. Hyperbolic Tangent Function**
- Hyperbolic Tanget function changes from -1,-1<F(x)<1 to 1.The formula of function is: **F(x)=[1-pow(e,-2x)]/[1+pow(e,-2x)]**

## Announcement
- Overview of Deep Learning, **Dimension Reduction** , **Model Selection** , **XGBoot** topics will be under **Deep Learning Repo** 
- **Convolutional Neural Networks (CNN)** will be under **Artificial Intelligence Repo (AI)** 
- **Computer Vision** , **Self Autonomous Driving** with Tensorflow-Keras & Computer Vision & Deep Learning Repos will be also shared 
- **Kubernates** repo will be also shared 
- You can also check out Java Spring Framework series which will include **Spring Core,Spring MVC,Spring Boot** repo under
[Java Spring Framework Repo](https://github.com/ozanyldzgithuboffical/Spring)
- You can also check out Machine Learning series which will include **Machine Learning Basis,Prediction,Pre-Processing,Regression,Classification,Clustring & Reinforcement Learning** techniques.
[Machine Learning Repo](https://github.com/ozanyldzgithuboffical/OzanYldzML)
- You can also check out Natural Language Processing (NLP) series.
[Natural Language Processing (NLP)](https://github.com/ozanyldzgithuboffical/NLP-Natural-Language-Processing-)
- **Computer Vision with Deep Learning** repo will be also available later.
- **Spring Microservices with Spring Cloud** repo will be also available later. 

## About the Repo
- This repo is open-source and aims at giving an overview about the top-latest topics that will lead learning of the basis of deep learning and intelligent systems basis.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate. Thanks.

**OZAN YILDIZ**
-Computer Engineer at HAVELSAN Ankara/Turkey 
**Linkedin**
[Ozan YILDIZ Linkedin](https://www.linkedin.com/in/ozan-yildiz-b8137a173/)

## License
[MIT](https://choosealicense.com/licenses/mit/)

