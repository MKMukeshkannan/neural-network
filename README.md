# NEURAL NETWORK IMPLEMENTATION in C
my attempt to build a neural network from scratch in c. 
NOTE: lot of thing are hardcoded and not extensible right now!!

# HOW TO RUN THE PROGRAM
```sh
clang src/main.c -o main && ./main
```
![SampleVideo](https://github.com/user-attachments/assets/2e7d72b5-f838-4d0f-87b4-8c569d454ed7)

# HOW DID I DO IT
### DATASTRUCTURE:
` typedef struct NeuralNetwork; `
primary data structure that contains all informations about the neural network.
weights and bias are persisted across epoch and m_neuron_layers and m_neuron_layers are temperory that gets 
reset every epoch.

### DATASET:
i took dataset used in this project from Kaggle, it has house price dataset that has 7 features and 1 output.
but i had to normalise them and split them into train and test split (90:10).

### STEPS:
1. read csv, in my case it reads from `data/train.csv`, load them as array.
2. create `NeuralNetwork` datastructure with required number of hidden layers and number of nodes in each layer.
3. randomize weights and bias in a normalised fashion.
4. now, ::train the model::
5. feed forward the neural network with `row_i`
6. now, do backward propogation, finding out how each weight in the neural neural affect the output.
7. adjust weights and biases accordingly.
8. repeat step 4,5,6,7 till MSE converges
