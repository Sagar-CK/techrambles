---
layout: post
title: "CI: Artificial Neural Networks (ANNs)"
date:   2024-08-20 11:57:46 +0200
categories: ai
usemathjax: true
---

If you've been following AI or tech news over the past few years, you've almost certainly come across the term **Artificial Neural Networks (ANNs).** ANNs have been gaining significant attention in mainstream media due to their role in prominent models like the transformer architecture-based GPT, known for its ability to generate human-like text.

While we won’t be (re)building [ChatGPT](https://chat.openai.com/chat) or any of its variants in today's article—though that could make for an interesting future topic 🤔—we will focus on understanding the key components in building such models: ANNs and their inner workings. As suggested by the term *Artificial* Neural Networks, these are biologically inspired AI models that train **neurons** within the network to map inputs to desired outputs. Much like the neurons in our brains, ANNs allow us to adapt, learn from new inputs, and make internal changes.

<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/neuron.png" alt="Simplified Representation of a Neuron">
    <p><strong>Simplified Representation of a Neuron</strong></p>
  </div>
</div>

As you might imagine, this is a fascinating yet complex topic. To make it easier to follow, we’ll be using the following roadmap in this article:
1. Introduction to Artificial Neurons and Perceptrons
2. The Multilayer Perceptron
3. Case Study: Training a Product Classification Robot

In our case study, we'll work with a grocery robot 🤖🛒 that initially has no idea what an apple is 💭. However, it can analyze a product’s features using its sensors. Based on attributes like shape, color, weight, and packaging, it can classify specific products. This problem can be simplified into **classification**, which an ANN can perform. Our task will be to design an ANN that predicts the correct product class for a given input ✅.

Throughout this process, we'll discuss how to select **hyperparameters** and **loss functions** for your model. Additionally, we'll cover topics like **regularization**, **learning rate**, and **batch size**, which can enhance the efficiency and effectiveness of your ANN.

With that said, let's get started!

### Introduction to Artificial Neurons and Perceptrons

#### Artificial Neuron

As mentioned earlier, artificial neurons share many similarities with human neurons. Our brain consists of a network of interconnected neurons. When a neuron receives a signal, it processes it and sends a signal to other neurons. This basic idea also applies to artificial neurons.

In 1943, Warren McCulloch and Walter Pitts introduced the first mathematical model of an artificial neuron. This model is based on the idea that a neuron can be represented as a binary threshold unit. The model takes a set of inputs, sums them up, and if the sum exceeds a certain threshold, the neuron fires and outputs a 1; otherwise, it outputs a 0. These inputs correspond to the values of the features in a dataset, and the output corresponds to the class of the data point. Later in our case study, we’ll see how this concept can be used to classify products.

<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/a_n1.png" alt="Simplified Representation of an Artificial Neuron">
    <p><strong>Simplified Representation of an Artificial Neuron</strong></p>
  </div>
</div>

Here, $$ x_1, x_2, ..., x_n $$ are the inputs to the neuron, $$ \theta $$ is the threshold, and $$ y $$ is the output. The summation can be represented as:

$$
\begin{equation}
 y = \begin{cases} 1, & \text{if } \sum_{i=1}^{n} x_i > \theta \\ 0, & \text{otherwise} \end{cases}
\end{equation}
$$

However, this artificial neuron would need to be either hard-coded or manually adjusted to classify specific inputs to outputs, limiting its ability to learn complex functions. To address this limitation, Frank Rosenblatt introduced the **Perceptron** in 1957, a simple model of a single-layer neural network.

#### Perceptron

<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/perceptron.png" alt="Perceptron Visualized">
    <p><strong>Perceptron Visualized</strong></p>
  </div>
</div>

Similar to an artificial neuron, the perceptron takes a set of inputs $$ x_1, x_2, ..., x_n $$ and corresponding weights $$ w_1, w_2, ..., w_n $$ and computes the weighted sum of the inputs. There is also a bias term $$ b $$ added to the sum, resulting in an intermediate value $$ z = w^T x + b $$. (The weighted sum between the inputs and weights can also be represented as the dot product of the weights and inputs or weights transposed times inputs). Consider the following example:

$$
\begin{equation}
 z = w^T x + b = w \cdot x + b
\end{equation}
$$

$$
 x = \begin{bmatrix} 0.2 \\ 1.7 \\ 2.1 \end{bmatrix}, w = \begin{bmatrix} 4.3 \\ 2.1 \\ 2.4 \end{bmatrix}, b = 1
$$

$$
 z = 4.3 \cdot 0.2 + 2.1 \cdot 1.7 + 2.4 \cdot 2.1 + 1 = 10.47
$$

Once $$ z $$ is computed, the perceptron applies an activation function $$ f $$ to the intermediate value $$ z $$ to get the output $$ y = f(z) $$. In this example, the activation function is a step function, which outputs 1 if the input is greater than 0 and 0 otherwise. This can be summarized as follows:

$$
\begin{equation}
 f(z) = \begin{cases} 1, & \text{if } z > 0 \\ 0, & \text{otherwise} \end{cases}
\end{equation}
$$

$$
\begin{equation}
 y = f(z)
\end{equation}
$$

To better understand this, let’s consider a simple example where we want to train a perceptron to predict the type of a property. The inputs could be the size of the property in square meters $$ (x_1) $$ and the number of rooms $$ (x_2) $$, with the output being the type of property ( $$ y = 0 $$ for an apartment and $$ y = 1 $$ for a house). We can represent the dataset as follows:

| Size m² $$ ( x_1 ) $$ | Rooms $$ ( x_2 ) $$ | Property Type $$ ( y) $$ |
| --------------------- | ------------------- | ------------------------ |
| 100                   | 3                   | 0                        |
| 120                   | 3                   | 1                        |
| 65                    | 2                   | 0                        |

To train the perceptron, we need to determine which weights and bias will correctly classify the inputs. While we could manually adjust the weights and bias, this would be inefficient. Instead, we use a **loss function** to measure the model's error and adjust the weights and bias to minimize this error. This process involves another hyperparameter called the **learning rate** $$ \alpha $$, a small positive number that determines how much the weights and bias should be adjusted in each iteration of the training process. Now, let's train our perceptron.

> **Steps for training a perceptron:**
> 1. Initialize the weights and bias to random values.
> 2. For each data point in the dataset, compute the perceptron’s output (inference).
> 3. Compute the loss using the loss function.
> 4. Update the weights and bias using the learning rate.

We will use the step function as the activation function and a simple loss function, which is the difference between the predicted and true output. The loss function can be written as:

$$ 
\begin{equation}
 \mathcal{L} = y - \hat{y}
\end{equation}
$$

where $$ y $$ is the true output, and $$ \hat{y} $$ is the predicted output. 

**1)** Initialize the weights and bias to random values. Let's say $$ w_1 = 0.5, w_2 = 0.5, b = -50, \alpha = 0.1$$.

**2)** Compute the inference for the first data point in the dataset. The input is $$ x = \begin{bmatrix} 100 \\ 3 \end{bmatrix} $$, the weights are $$ w = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} $$, and the bias is $$ b = -50 $$.

$$
z = w^T x + b = 0.5 \cdot 100 + 0.5 \cdot 3 - 

50 = 1.5
$$

$$ 
\hat{y} = f(z) = 1
$$

**3)** Compute the loss:

$$ 
\mathcal{L} = y - \hat{y} = 0 - 1 = -1
$$

**4)** Update the weights and bias using the learning rate $$ \alpha = 0.1 $$:

$$
\Delta w_1 = \alpha \cdot \mathcal{L} \cdot x_1 = 0.1 \cdot (-1) \cdot 100 = -10
$$

$$ 
\Delta w_2 = \alpha \cdot \mathcal{L} \cdot x_2 = 0.1 \cdot (-1) \cdot 3 = -0.3
$$

$$ 
\Delta b = \alpha \cdot \mathcal{L} = 0.1 \cdot (-1) = -0.1
$$

Update the weights and bias:

$$
w_1 = w_1 + \Delta w_1 = 0.5 - 10 = -9.5
$$

$$
w_2 = w_2 + \Delta w_2 = 0.5 - 0.3 = 0.2
$$

$$
b = b + \Delta b = -50 - 0.1 = -50.1
$$

By doing this we have effectively shifted the weights and bias in the direction that minimizes the error.  We repeat this process for the remaining data points in the dataset. With each iteration, the perceptron updates the weights and bias to minimize the loss, gradually improving its classification accuracy 🙌. 

At this point, you might have asked yourself why do we need weights **AND** bias in the first place? Why not just weights? This is because the weights determine how much each **input contributes to the prediction** while the bias determines the **threshold at which the perceptron fires**. Hence, by adjusting the weights we can change the **slope** of the decision boundary, while by adjusting the bias we can change the **intercept** of the decision boundary. This can be visualized as follows:

<div class="about-container">
  <div class="free-item">
    <img src="/assets/images/ann/weights_bias.png" alt="Weights and Bias in a Perceptron" >
    <p><strong>Weights and Bias in a Perceptron</strong></p>
  </div>
</div>

Amazing! We now know how to train a perceptron. However, as you can imagine, the perceptron is quite limited in its capabilities. It can only classify **linearly separable** data points, which means that it can only classify data points that can be separated by a straight line. Even the most basic datasets can be quite complex and not linearly separable. The XOR dataset is a classic example of this. The XOR dataset consists of four data points, two of which belong to class 0 and two of which belong to class 1. The data, although simple, cannot be separated by a straight line. 

<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/xor.png" alt="XOR Dataset with Decision Boundary">
    <p><strong>XOR Dataset with Decision Boundary</strong></p>
</div>
</div>

In order to classify such data points, we need a more complex model. This is where the **Multilayer Perceptron (MLP)** comes in. 

### The Multilayer Perceptron

### Case Study: Training a Product Classification Robot!


### Sources and External Links for Further Reading