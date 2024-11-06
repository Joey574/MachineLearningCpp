# MachineLearningC++
## Solution for me to experiment and test various kinds of neural networks and simulate cool fractals
This project contains various different kinds of neural networks and datasets. Code is written more or less from scratch, from the matrix class to dot prod.

## Datasets
Some of the datasets / fractals included are
* **MNIST**
* **Mandlebrot**
* **FMNIST**

With future plans for datasets being
* **Barnsley ferns**
* **N-body simulation data**

Dataset loaders are just stored in a static class in a .cpp file, easy enough to include. The actual datasets are located in **_Dependencies\datasets_**
<br>

## Neural Networks
There are many different iterations of neural networks in the solution, these are built as .lib files and stored in **_Dependencies\lib_** the following are the names of the aforementioned neural network projects.
* **NeuralNetwork**
* **NeuralNetwork_2**
* **SingleBlockNeuralNetwork**
* **SingleBlockCudaNetwork**

### Neural Network
This was my first attempt at a neural network class in c++, I've made function-based networks in the past, although these relied heavily on global variables and c++ malpractice. This was my first attempt to move away from that and standardize my use of neural networks and machine learning techniques.
<br><br>
This network makes use of the **Matrix** class to do most of the arithmetic, and doesn't implement too many optimizations in of itself, some of the ones it does include being **Bias.h** to handle calculations regarding biases, keeping them all in one location in memory. Also uses a custom update loop for weights and biases that takes better advantage of simd intrinsics.
<br><br>
Major performance problems with this network are mostly tied to the matrix class itself, which I'll go more in depth in a dedicated section, however, just know, theres's a lot of unnecessary memory allocations and copies. *Who doesn't love temp objects?* 
<br><br>
The higher level structure I created for this class set the basis for all other classes, although exact implementations vary, and different networks have various levels of abstraction, the general control flow looks the same for all of them.

```mermaid
flowchart TD
a([User]) --> d{{Define}} & c{{Compile}} & Fit
Fit --> fp{{Forward Prop}} & bp{{Back Prop}}
```

```mermaid
flowchart TD
subgraph ForwardProp
fpa{{Compute total values}} --> fpb{{Apply activation function}}
end

subgraph BackProp
bpa{{Compute loss}} --> bpb{{Compute gradient}} --> bpc{{Update weights and biases}}
end

subgraph Define
da{{Set dimensions, sizes, and activation data}}
end

subgraph Compile
ca{{Set pointers to loss and activation functions}} --> cb{{Allocate and initialize weights and biases}}
end
```

<br>

### Neural Network 2
**CURRENTLY NOT COMPLETE** (or even started truthfully) <br>
This project aims to take the organization I learned from NeuralNetwork and lobotomize it with template metaprogramming and other eldritch techniques

### Single-Block Neural Network
This project completely does away with the **Matrix** class I poured my soul into, instead it opts for the ever loved pointers. Specifically, I allocate memory for three different pointers
* *m_network*
* *m_batch_data*
* *m_test_data*

which are structured as follows 
<br>

```mermaid
block-beta
    block
        columns 1
        a(["Network"]):1
        b["Weights"]:1
        c["Biases"]:1
    end
    block
        columns 1
        d(["Batch Data"]):1
        e["Total"]:1
        f["Activation"]:1
        g["Total Derivatives"]:1
        h["Weight Derivatives"]:1
        i["Bias Derivatives"]:1
    end
block
        columns 1
        j(["Test Data"]):1
        k["Total"]:1
        l["Activation"]:1
    end
```

I also make heavy use of pointer arithmetic to make my life easier, such as *m_biases* which points to **_biases_**, *m_test_activation* which points to... you guessed it, **_test_activation_**, and so on so forth.
<br><br>
The main benefit of this network over the **NeuralNetwork** class is that it allocates all the memory it needs up front. By doing this, we avoid the creation of temp objects and massive copies, instead just storing the data directly where we want it to be in the first place. Another benefit of doing this is we can chain operations together much better, for exmaple, if we wanted to do **_A = B * C + D_** where A, B, C, D are all matrices of the same size, in the **Matrix** class that would look something like this

```mermaid
    flowchart TD
    a(Create temp1) --> b(Store B * C in temp1)
    b --> c(Create temp2) -->d(Store temp1 + D in temp2)
    d --> e(Copy temp2 to A)
```

Quite a lot of work, and most of it is pointless too. Instead, if we apply a minimal amount of thinking, we can do this.

```mermaid
    flowchart TD
    a(Store the result in A)
```

Woah! That's insane! As it turns out, if we know we want the data to be in A from the beginning, we can just directly store data into A, on top of that, we can directly chain the simd operations we want to do together, doing them all in place, meaning we don't need any temp objects, and in this specific situation we can directly compute **_B * C + D_** with a fmadd simd instrinsic leaving us with just a few instructions, much better than all the temp objects and copies we were previously doing.
<br><br>
Transposes are also a big part of forward_prop and back_prop, however, these transposes are only really used in dot products with other matrices, as such, if we just *read* the data in a transposed manner, we can get the right result, without having to bother with actually transposing the matrix. Changing around a couple for loops also ensures we maintain decent cache access.
<br><br>
Of course, this class also make use of optimizations already present in the **Matrix** class, like omp for parallelization, and simd intrinsics.
<br><br>
Overall I observed a 3-4x performance boost using this class over the **NeuralNetwork** class
<br>
Below are a couple results on various network sizes for **MNIST**
<br><br>
![Performance Tests](https://github.com/Joey574/MLImageLearning/blob/main/Mandlebrot%20Aproximations/misc/Screenshot%202024-11-05%20192720.png)

### Single-Block Cuda Network
**CURRENTLY NOT WORKING** <br>
This project is my first attempt at running a neural network completely on CUDA, I hope to use a similair design to **SingleBlockNeuralNetwork**


## Actual Projects
Ok so now we've convered the stuff that does all the math, I'd like to mention the stuff that actually does stuff, specifically
* **MNISTNetwork**
* **MNIST_SBNeuralNetwork**
* **MNIST_CNN**
* **MandlebrotNetwork**
* **Mandlebrot_SBNN**
<br>

### MNISTNetwork
This network trains on the **MNIST** dataset using the **NeuralNetwork** class, truly machine learning 101, nothing all that special to say about it.
<br>

### MNIST_SBNeuralNetwork
This one, while quite a mouthful, trains on the **MNIST** dataset, using the **SingleBlockNeuralNetwork** class, this project currently holds my record on mnist of **98.47%** which was achieved using 3 hidden layers of size 1024, and leaky_relu activation in the hidden layers, and sigmoid for the final.
<br>

### MNIST_CNN
This network trains on, yet again, the **MNIST** dataset using the **SingleBlockCudaNetwork** class, although at time of writing, is still not functional.
<br>

### MandlebrotNetwork
This network is what really got me into fractals and neural networks alike, taking inspiration from 
<br>
https://www.youtube.com/watch?v=TkwXa7Cvfr8 (great video by the way you should totally watch it)
<br><br>
I took the idea of approximating the mandlebrot with neural networks and ran with it, this specific version uses the **NeuralNetwork** class, some of the better images it produced being
<br><br>

![Mandlebrot approximation](https://github.com/Joey574/MLImageLearning/blob/main/Mandlebrot%20Aproximations/4-27-24%20Big%20Network%202/5_9_24_final(19).bmp)
*This bad boy was trained on and off over the course of a couple weeks* <br><br>

![Mandlebrot approximation](https://github.com/Joey574/MLImageLearning/blob/main/Mandlebrot%20Aproximations/4-25-24%20Big%20Network%201/4_27_24_epoch1.bmp)
*This guy probably took closer to a month, although it used a smaller network than in the previous image*<br><br>

*Oh and just for reference heres what a "perfect" mandlebrot would look like* <br>
![Perfect mandlebrot](https://github.com/Joey574/MLImageLearning/blob/main/Mandlebrot%20Aproximations/PerfectMandlebrots/1920_1080_500_0.95.bmp)

### Mandlebrot_SBNN
Thinking about how much time I spent using the old network pains me, with the performance improvements I managed with the **SingleBlockNeuralNetwork** perhaps I could've actually touched grass. At any rate this uses the aforementioned network and approximates the mandlebrot. 
<br><br>
As I finished this one much more recently, and I kind of exhausted myself on the mandlebrot already, I haven't trained with this one anywhere near as much, as such predictions are much lower quality.

![Mandlebrot lesser approximation](https://github.com/Joey574/MLImageLearning/blob/main/Mandlebrot%20Aproximations/9-12-24%20Desktop/big%20net%2C%20new%20system/final.bmp)

## Honorable Mentions

### Matrix
Truly this is where it began
