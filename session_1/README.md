The goal of this assignment is to build a small neural network from scratch. The assignment is based on micrograd by Andrej Karpathy.

Assignment setup:
- Install graphviz: `pip install graphviz`
- Install the graphviz package in your system: `sudo apt-get install graphviz` (for Ubuntu) or `brew install graphviz` (for mac)

Exercises:

1. Fill in the Value class to enable the lol() function. You should get -4.0 as a result.
2. Use the lol_grad() function below to manually compute dL/df, dL/dd, dL/de, dL/dc, dL/da, dL/db
3. Implement the function tanh() in the Value class and build the following network
4. Implement the backward() function in each Value's operator: __add__, __mul__, __tanh__ and assign it to _backward
5. Why are we using self.grad += and not self.grad = in the backward() functions?  Answer as a comment in this cell.
6. Replace the tanh() function by a combination of exp() and division.  Implement the __exp__ and __pow__ functions in Value. You should get the same results as before.
7. Let's build an MLP based on the Value class.  Fill in the Neuron, Layer and MLP classes.
8. Write a training loop
9. Why do you need to reset the grads to zero at every training step?  Answer as a comment in this cell.
10. Try to fit the x^2 function with a three-layer MLP. Why is it so hard?
