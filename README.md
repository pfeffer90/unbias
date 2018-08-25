# unbias
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
 
 
