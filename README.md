# unbias

## Jupyter Hub for online experiments

Follow [this link](http://142.93.247.119) to log on the jupyter hub and participate in the unbias experiments.

## Install 
```
conda env create -f environment.yml
```
## Run tests
```
make test
```

## Task diary

### Generate artificial decision data from a stationary agent

Assume an agent with fixed $b \in R$ and $w \in R^N$, where $N$ is the memory strength. Then generate binary decisions via
 $$
 p(x_t = 1|x_t-1, ..., x_t-N) = \sigma(b+wx)
 $$

[Original Paper on the Outguessing Machine : (http://this1that1whatever.com/miscellany/mind-reader/Hagelbarger-SEER.pdf)]

[ Shannon Paper : http://www.this1that1whatever.com/miscellany/mind-reader/Shannon-Mind-Reading.pdf]

[Spontaneous flight manouveurs in flies: http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000443]

 ### Look for algorithms to infer b and w for a stationary agent
 
 * Bayesian inference: We divide the decision into pairs of current decision and history $(x_t, x_t-1)$ and try to calculate the posterior $p(b,w|{x_t, x_t-1})$
 * logistic regression: gradient descent on the log likelihood $p(D|w)$
 
 ### Write a skeleton for the backbone of the game
 
 Model of the game
 * acquire next agent choice
 * provide a reward 
 * decide whether game continues
 
 ### Get a jupyter notebook running with user interaction 
 
 ### Run the jupyter notebook game on a jupyter hub server to allow online data collection
 
 ### Run first experiment
 
 * two conditions: without feedback and with feedback to test whether participants with feedback are less biased (maybe it is more interesting to just run without feedback to have a free test of biases, also more data, then run the other one on Tuesday)
 * what do we want to analyse
     - indiviudal bias and serial correlation structure
     - performance of the outguesser (this is best done in the no feedback case, because then interactions with the algo are excluded, we should be able to run other outguessers against this data)
 * what is missing
     - make small self-generated data set for real life test
     - check whether data structure is enough and allows the above analysis, try with self-generated data
     - set up jupyter hub accounts for the people to play
     - think of how many trials/rounds we need in each of the two experiments
     - appropriate choice of learning rate and convergence check of the gradient descent
     - problem with saving model parameters for history length larger than one
     
 
 
