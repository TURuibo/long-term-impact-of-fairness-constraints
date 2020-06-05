# long-term-impact-of-fairness-constraints
This repository can be used for analysing the long-term impact of static fairness constraints.  
- The functions and classes for our model are in "fico_util.py", "util.py", and "eq_odds.py".  
- We present the applications of our model with Jupyter notebook.  
- The figure in the FICO experiment is plotted by Matlab. The code is in the folder "FICO_fig".  

## Installation
Step 1: Install [Python 3](https://www.python.org/downloads/).  

Step 2: Intall [pip](https://pip.pypa.io/en/stable/installing/).  

```curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py```  
```python get-pip.py```   

Step 3: Install [Jupyter notebook](https://jupyter.org/install).  
```pip install notebook```

Step 4: Install the following packages.  fairlearn, tempeh, numpy, pandas, scipy, sklearn, matplotlib, cvxpy, pynverse, random, copy, progressbar, collections  
```pip install XXX``` (replace XXX with the name of packages)  

Step 5: Restart the Jupyter Notebook and run the code.  

## notebooks
The synthetic data, FICO, and COMPAS experiments are shown in the folder notebook.  

## COMPAS_Fig
Figures and results of COMPAS experiment.  

## FICO_fig
Figures and results of FICO experiment.   

## data
The simulated FICO data.  