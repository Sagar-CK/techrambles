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
> 1. Introduction to Artificial Neurons and Perceptrons
> 2. The Multilayer Perceptron
> 3. Case Study: Training a Product Classification Robot

In our case study, we will build a grocery robot (ANN) 🤖🛒, which initially has no idea what an apple is, into one that can identify them. Our robot can analyze a product’s features using its sensors like shape, color, weight, and packaging. Using this information we want to create a classify a product from its features (inputs) to its product type (output/class). This problem can be simplified is also known as **classification**.

Through this process, we'll discuss how to select **hyperparameters** and **loss functions** for your ANN. Additionally, we'll cover topics like **regularization**, **learning rate**, and **batch size**, which can enhance the efficiency and effectiveness of your ANN.

With that said, let's get started!

### Introduction to Artificial Neurons and Perceptrons

#### Artificial Neuron

As previously mentioned, artificial neurons share many similarities with human neurons. Our brain consists of a network of interconnected neurons. When a neuron receives a signal, it processes it and sends a signal to other neurons. This basic idea also applies to artificial neurons.

In 1943, [Warren McCulloch and Walter Pitts](https://www.historyofinformation.com/detail.php?entryid=782) introduced the first mathematical model of an artificial neuron. This model is based on the idea that a neuron can be represented as a binary threshold unit. The model takes a set of inputs, sums them up, and if the sum exceeds a certain threshold, the neuron fires and outputs a 1; otherwise, it outputs a 0. 

These inputs correspond to the **values of the features** in a dataset, and the **output corresponds to the class** of the data point. Later in our case study, we’ll see how this concept can be used to classify products.

<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/a_n1.png" alt="Simplified Representation of an Artificial Neuron">
    <p><strong>Simplified Representation of an Artificial Neuron</strong></p>
  </div>
</div>

Here, $$ x_1, x_2, ..., x_n $$ are the inputs to the neuron, $$ \theta $$ is the threshold, and $$ y $$ is the output (you can imagine 1 and 0 represent different classes). The neuron can be represented as:

$$
\begin{equation}
 y = \begin{cases} 1, & \text{if } \sum_{i=1}^{n} x_i > \theta \\ 0, & \text{otherwise} \end{cases}
\end{equation}
$$

However, this artificial neuron would need to be either hard-coded or manually adjusted to classify specific inputs to outputs (due to the $$ \theta $$ threshold), limiting its ability to learn complex functions by itself. To address this limitation, [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) introduced the **Perceptron** in 1957, a simple model of a single-layer neural network.

#### Perceptron

<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/perceptron.png" alt="Perceptron Visualized">
    <p><strong>Perceptron Visualized</strong></p>
  </div>
</div>

Similar to how an artificial neuron functions, a perceptron processes a set of inputs -- denoted as $$ x_1, x_2, \dots, x_n $$ -- each paired with a corresponding weight $$ w_1, w_2, \dots, w_n $$. It calculates a weighted sum of these inputs. Additionally, a bias term $$ b $$ is included in this sum, resulting in an intermediate value $$ z = w^T x + b $$. (Here, the weighted sum of inputs and weights can also be expressed as the dot product of the weights and inputs, or as the transpose of the weights multiplied by the inputs). Consider the following example:

$$
\begin{equation}
 z = \vec{w}^T \vec{x} + b = \vec{w} \cdot \vec{x} + b
\end{equation}
$$

$$
 x = \begin{bmatrix} 0.2 \\ 1.7 \\ 2.1 \end{bmatrix}, w = \begin{bmatrix} 4.3 \\ 2.1 \\ 2.4 \end{bmatrix}, b = 1
$$

$$
 z = 4.3 \cdot 0.2 + 2.1 \cdot 1.7 + 2.4 \cdot 2.1 + 1 = 10.47
$$

Note, for clarity of the dot product and transpose, we have used a arrow notation to represent vectors. Often this is left out in practice and variables are inferred to be vectors.

Once $$ z $$ is computed, the perceptron then applies an activation function $$ f $$ to the intermediate value $$ z $$ to get the output $$ y = f(z) $$. In this example, the activation function is a [step function](https://www.geeksforgeeks.org/activation-functions/), which outputs 1 if the input is greater than or equal to 0 and 0 otherwise. This can be summarized as follows:

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

To better understand this, let’s consider a simple example where we want to train a perceptron to predict the type of a property. The inputs could be the size of the property in square meters $$ x_1 $$ and the number of rooms $$ x_2 $$, with the output being the type of property  $$ y = 0 $$ for an apartment and $$ y = 1 $$ for a house. We can represent the dataset as follows:

| Size m² $$ ( x_1 ) $$ | Rooms $$ ( x_2 ) $$ | Property Type $$ ( y) $$ |
| --------------------- | ------------------- | ------------------------ |
| 100                   | 3                   | 0                        |
| 120                   | 3                   | 1                        |
| 65                    | 2                   | 0                        |

To train the perceptron, we need to determine which weights and bias will (best) correctly classify the inputs. While we could manually adjust the weights and bias, this would be **very inefficient**. 

Instead, we can use a [**loss function**](https://en.wikipedia.org/wiki/Loss_function) to measure the error between the model's predictions and the actual outputs, and then adjust the weights and bias to minimize this error. This adjustment process involves a hyperparameter called the [**learning rate**](https://www.purestorage.com/knowledge/what-is-learning-rate.html), denoted as $$ \alpha $$, which is a small positive number that controls how much the weights and bias are updated during each iteration of the training process (to prevent drastic changes in weights). Now, let's train our perceptron.

> **Steps for training a perceptron:**
> 1. Initialize the weights and bias to random values. Select a learning rate.
> 2. For each data point in the dataset, compute the perceptron’s output (inference).
> 3. Compute the loss using the loss function.
> 4. Update the weights and bias using the learning rate.

Following the previous example we will use the step function as the activation function. Also, we will use a simple loss function: the difference between the predicted and true output. The loss function can be written as:

$$ 
\begin{equation}
 \mathcal{L} = y - \hat{y}
\end{equation}
$$

where $$ y $$ is the true output, and $$ \hat{y} $$ is the predicted output. 

**1)** Initialize the weights and bias to random values. Let's say $$ w_1 = 0.5, w_2 = 0.5, b = -50$. Choose a learning rate $$ \alpha = 0.1 $$.

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

By doing this, we have effectively adjusted the weights and bias in the direction that minimizes the error. We repeat this process for each data point in the dataset. With every iteration, the perceptron updates the weights and bias to reduce the loss, gradually enhancing its classification accuracy 🙌. (Still skeptical? Continue reading until the case study section, where we'll write code to verify this).

At this point, you might wonder why we need both weights **and** bias. Why not just weights? The reason is that weights determine how much each **input contributes to the prediction**, while the bias determines the **threshold at which the perceptron activates**. Therefore, adjusting the weights changes the **slope** of the decision boundary, while adjusting the bias alters the **intercept** of the decision boundary. This can be visualized in the two dimensional case as follows:

<div class="about-container">
  <div class="free-item">
    <img src="/assets/images/ann/weights_bias.png" alt="Weights and Bias in a Perceptron" >
    <p><strong>Weights and Bias in a Perceptron</strong></p>
  </div>
</div>

Great! We now understand how to train a perceptron. However, as you might have guessed, the perceptron has its limitations. One significant limitation is that it can only classify **linearly separable** data points, meaning it can only distinguish between data points that can be separated by a straight line. In reality, even the simplest datasets can be more complex and not linearly separable. A classic example of this is the XOR dataset, which consists of four data points—two belonging to class 0 and two to class 1. Despite its simplicity, this data cannot be separated by a straight line.

<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/xor.png" alt="XOR Dataset with Decision Boundary">
    <p><strong>XOR Dataset with Decision Boundary</strong></p>
</div>
</div>

In order to classify such data points, we will need a more complex infrastructure. This is where the **Multilayer Perceptron (MLP)** comes in. 

### The Multilayer Perceptron

<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/mlp.png" alt="Multilayer Perceptron">
    <p><strong>Multilayer Perceptron</strong></p>
</div>
</div>

The [Multilayer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) is a type of artificial neural network that consists of multiple layers of neurons. Each layer is fully connected to the next layer, meaning **each neuron in one layer is connected to every neuron in the next layer**. The first layer is the input layer, the last layer is the output layer, and the layers in between are called hidden layers (in our example we just have on hidden layer). Simply described, **the MLP is a stack of perceptrons**, where the output of one perceptron is the input to the next perceptron.

The neurons in the hidden layers use an activation function to transform the weighted sum of their inputs. Finally, the output of the MLP is determined by the output of the neurons in the output layer. Note, we can have multiple neurons in the output layer, depending on the number of classes we want to classify (e.g 3 neurons for 3 classes).

In a multilayer perceptron, the activation function is typically referred to as $$ \sigma $$, and are often non-linear functions. This is because the use of non-linear activation functions allows the MLP to learn complex patterns in the data, specifically nonlinearities 🤯. Some common activation functions include the [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [tanh](https://en.wikipedia.org/wiki/Hyperbolic_functions), and [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) functions.


<div class="about-container">
  <div class="free-item">
    <img src="/assets/images/ann/activations.png" alt="Common Activation Functions">
    <p><strong>Common Activation Functions</strong></p>
</div>
</div>

Additionally, smooth (differentiable) activation functions are preferred as they are suited to perform [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), an optimization algorithm used to minimize the loss function (what we did in the perceptron example, but now with multiple layers and a lot more math 😅).First, we will go over what the "forward" propogation (inference) looks like for this network.

<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/mlp_overview.png" alt="MLP Network Overview">
    <p><strong>MLP Network Overview</strong></p>
</div>
</div>

In this case we have an input layer with four features $$ x_1, x_2, x_3, x_4 $$, a hidden layer with three neurons $$ h_1, h_2, h_3 $$, and an output layer with two neurons $$ \hat{y}_1, \hat{y}_2 $$. 

$$

\begin{equation}
\begin{aligned}
x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix}, 
h = \begin{bmatrix} h_1 \\ h_2 \\ h_3 \end{bmatrix},
\hat{y} = \begin{bmatrix} \hat{y}_1 \\ \hat{y}_2 \end{bmatrix}
\end{aligned}
\end{equation}

$$


The weights are denoted on a layer level by $$ W^{i} $$, where $$ i $$ is the layer number. Each individual layer weight is denoted by $$ w^{i}_{j,k} $$, where $$ i $$ is the layer number, $$ j $$ is the neuron in the current layer, and $$ k $$ is the neuron in the next layer. The bias is denoted by $$ b^{i} $$, where $$ i $$ is the layer number. Note, since we have multiple dimensions our bias will also be a vector.

$$

\begin{equation}
\begin{aligned}
W^{1} = \begin{bmatrix} w^{1}_{1,1} & w^{1}_{1,2} & w^{1}_{1,3} \\ w^{1}_{2,1} & w^{1}_{2,2} & w^{1}_{2,3} \\ w^{1}_{3,1} & w^{1}_{3,2} & w^{1}_{3,3} \\ w^{1}_{4,1} & w^{1}_{4,2} & w^{1}_{4,3} \end{bmatrix},

b^{1} = \begin{bmatrix} b^{1}_{1} \\ b^{1}_{2} \\ b^{1}_{3} \end{bmatrix}

\end{aligned}
\end{equation}

$$

$$

\begin{equation}
\begin{aligned}
W^{2} = \begin{bmatrix} w^{2}_{1,1} & w^{2}_{1,2} \\ w^{2}_{2,1} & w^{2}_{2,2} \\ w^{2}_{3,1} & w^{2}_{3,2} \end{bmatrix},

b^{2} = \begin{bmatrix} b^{2}_{1} \\ b^{2}_{2} \end{bmatrix}

\end{aligned}
\end{equation}

$$

Note, the notations we use here for the hidden layer and output layer are just for clarity. In practice, the hidden layer and output layer can have any number of neurons and notation can vary depending on the context.

With all of this infomration we can now compute our intermediate values and outputs for the network. Note, $$ \sigma $$ is the activation function, which can be any of the functions mentioned above. Additionally, $$ z^{(i)}_j $$ is the intermediate value for neuron $$ j $$ in layer $$ i $$. For the hidden layer, the inputs would be the intermediate values from the input layer, and for the output layer, the inputs would be the intermediate values from the hidden layer. We can write the equations for the hidden layer as follows:

$$
\begin{equation}
\begin{aligned}
z^{1} = W^{1} \cdot x + b^{1} \\
h = \sigma(z^{1}) = \begin{bmatrix} \sigma(z^{1}_{1}) \\ \sigma(z^{1}_{2}) \\ \sigma(z^{1}_{3}) \end{bmatrix} \\
\end{aligned}
\end{equation}
$$

Similarly, we can write the equations for the output layer as follows:

$$
\begin{equation}
\begin{aligned}
z^{2} = W^{2} \cdot h + b^{2} \\
\hat{y} = \sigma(z^{2}) = \begin{bmatrix} \sigma(z^{2}_{1}) \\ \sigma(z^{2}_{2}) \end{bmatrix} \\
\end{aligned}
\end{equation}
$$

Overall we can write the forward propogation as:

$$
\begin{equation}
\begin{aligned}
\hat{y} = \sigma(W^{2} \cdot h + b^{2}) = \sigma(W^{2} \cdot \sigma(W^{1} \cdot x + b^{1}) + b^{2})
\end{aligned}
\end{equation}
$$

And that is the forward propogation of the MLP! Now that we have the output, we can compute the loss and use gradient descent to update the weights and biases. This process is known as **backpropogation**, and is essentially minimizing the loss function by adjusting the weights and biases in the network as previously discussed.

<div class="about-container">
  <div class="free-item">
    <img src="/assets/images/ann/backprop.png" alt="Backpropogation in a Neural Network">
    <p><strong>Backpropogation in a Neural Network</strong></p>
</div>
</div>

Since the math is **MUCH** more involved, we will not cover it in this article. However, if you are interested in learning more about backpropogation, we recommend checking out this [video](https://www.youtube.com/watch?v=Ilg3gGewQ5U) by 3Blue1Brown and we will cover it in a future article.

In essence, gradient descent (which relies on backpropogation) is used to minimize the loss function by adjusting the weights and biases in the network. This process is repeated for multiple iterations until the loss converges to a minimum value. Once the loss converges, the network is trained and can be used to make predictions on new data. The example in the image above is the MSE loss function, which is commonly used for regression problems.

$$
\begin{equation}
\begin{aligned}
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\end{aligned}
\end{equation}
$$

The loss function is dependent on correct outputs $$ y_i $$ and predicted outputs $$ \hat{y}_i $$, which in turn depend on the inputs $$ x_i $$ and the weights and biases in the network. The goal of training the network is to minimize this loss function by adjusting the weights and biases. In most cases, the only element we can control is the weights and biases.

Thus, we search for the optimal weights and biases that minimize the loss function $$ \mathcal{L} $$ for all training data points. Finding the global minimum analytically is computationally unfeasible. Instead, we update the weights step by step until we find a set of weights that minimizes $$ \mathcal{L} $$, using the gradient descent update:

$$
\begin{equation}
\begin{aligned}
w_{new} = w_{old} - \alpha \frac{\partial \mathcal{L(w)}}{\partial w} | _{w = w_{old}}
\end{aligned}
\end{equation}
$$

This update rule simply means that we adjust the weights in the direction that minimizes the loss function -- consistent with the perceptron example. It can be thought of as a hill-climbing algorithm, where we adjust the weights in the direction that minimizes the loss function. The learning rate $$ \alpha $$ controls the size of the step we take in this direction. If the learning rate is too large, we might overshoot the minimum, and if it is too small, we might take too long to converge. This can be visualized as follows:

<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/gradient_descent.png" alt="Gradient Descent">
    <p><strong>Gradient Descent</strong></p>
</div>
</div>

For those who are interested (and have a strong mathematical background), we recommend checking out the [backpropogation](https://en.wikipedia.org/wiki/Backpropagation) and [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) Wikipedia pages for more information. Here is an overview of the math involved in backpropogation, if you were to derive it for a MSE loss function:

<div class="about-container">
  <div class="free-item">
    <img src="/assets/images/ann/gradient_formula.png" alt="Gradient Descent Formulae">
    <p><strong>Gradient Descent Formulae</strong></p>
</div>
</div>

Overall, the training procedure for an MLP can be summarized as follows:
<div class="about-container">
  <div class="basic-item">
    <img src="/assets/images/ann/mlp_training.png" alt="MLP Training Procedure">
    <p><strong>MLP Training Procedure</strong></p>
</div>
</div>

Phew! That was a lot of information. We hope you now have a better understanding of how artificial neural networks work, and how they can be trained to classify data points (if you have any lingering questions, feel free to engage with us at [techramblesblog@gmail.com](mailto:techramblesblog@gmail.com)). To solidify this knowledge, we will now apply these concepts in a case study where we train a product classification robot 🤖🛒!

### Case Study: Training a Product Classification Robot!


### Sources and External Links for Further Reading