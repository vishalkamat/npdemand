# npdemand: Python package for nonparametric analysis of counterfactual demand

This Python package implements the tools in [Kamat and Norris (2022)](https://arxiv.org/abs/2002.00103) to evaluate the average willingness to for a price decrease as well as the effect on demand of a general price change in the presence of exogenous, discrete variation of prices. It allows for three specification of demand: (i) nonparametric demand with only assuming demand for an alternative increases with the price of other alternatives; (ii) specification (i) along with additionally imposing that demand is separable; and (iii) specfication (ii) along with additionally imposing that demand is parameterized to be polynomial of degree K in each price.

# Contributors

Vishal Kamat, Toulouse School of Economics

Samuel Norris, University of British Columbia

# Installation

Store function in same directory as main file and import the function by copy pasting the following at the top of the file:

    from npdemand import *

Note that usage of the function requires the following to be already installed in python: Gurobi, sympy, and numpy.

# Implementation

**Syntax:**

    npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec="NPB",K="3",grid_size=5,conf=0,n=0,share_b={},Sigma="bs",level=0.95,incr="auto")

**Arguments:**

 - p_a       : Numpy array of size J (which is the number of alternatives) containing higher prices in price decrease. 
 - p_b       : Numpy array of size J containing lower prices in price decrease. 
 - P_obs     : List of numpy arrays of size J containing the observed discrete variation. 
 - share     : List of numpy arrays of size J containing the shares at each observed price. 
 - g_a       : Numpy array of size J containing the weight for demand at price p_a in parameter of interest. 
 - g_b       : Numpy array of size J containing the weight for demand at price p_b in parameter of interest. 
 - g_ab      : Numeric value corresponding to the weight for the average willingness to pay for price decrease from p_a to p_b. Set to 0 if p_b is not smaller than p_a.
 - spec      : Specification equal to "NPB" (specification (i) above), "NPS" (specification (ii) above), or "PS" (specification (iii) above). (*Optional*, default = "NPB")
 - K         : Degree of polynomial if spec = "PS". (*Optional*, default = 3)
 - grid_size : Size of gequidistant grid between p_lower and p_upper at which parameter restrictions are evluated. (*Optional*, default = 5)
 - conf      : If equal to 1 then construct confidence intervals. (*Optional*, default = 0)
 - n         : sample size (*Not optional* if conf = 1)
 - share_b   : Dictionary where each entry is a share computed in bootstrap draw (*Not optional* if conf = 1)
 - Sigma     : Estimate of var/covaraince matrix of shares. (*Optional*, default = "bs" which computes using bootstrap, i.e. using share and share_b)
 - level     : Level for confidence interval (*Optional*, default = 0.90)
 - incr      : Increment to use in test inversion procedure for confidence interval construction (*Optional*, default = "auto: which constructs using standard deviation of bootstrap bound estimates)

**Output:**

Stored as numpy array, say output, where: 
 - output[0] = lower bound estimate
 - output[1] = upper bound estimate
 - output[2] = lower value of confidence interval
 - output[3] - upper value of confidence interval

# Comments and Issues

Please post issues and comments in [Issues](https://github.com/vishalkamat/npdemand/issues).

# Citation

If you use the package, please cite the original paper [Kamat and Norris (2022)](https://arxiv.org/abs/2002.00103).
