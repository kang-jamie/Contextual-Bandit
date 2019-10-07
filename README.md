# High-Dimensional Nonlinear Contextual Bandits
or simply, Online Data-Informed Decision Makings

## Short intro
We study contextual multi-armed bandit algorithms in the setting where the expected rewards depend nonlinearly on a high-dimensional context vector. For simplicity, we focus on the 2-armed bandit model, but our work can be extended to more than two arms. We use nonparametric methods and compare their performance with the literature models.

## Long intro 
In many real world settings, optimal decisions might depend on the agent's specific characteristics (i.e. covariates) such that "personalized" decision-making might be preferred to the "one-size-fits-all" types of decisions. For instance, in a medical trial setting, whether the new drug might be good or not would highly depend on the patients' own chracteristics (age, BMI, disease status, ...). This has motivated the recent interests in contextual bandit problems. For more background, see https://banditalgs.com/. Some of these works include Bastani and Bayati (2015) (http://web.stanford.edu/~bayati/papers/lassoBandit.pdf), which uses LASSO estimator to tacke high-dimensionality that arises often in big data settings. 

We extend this result by using several nonparametric methods from Machine Learning, including MARS. Our method is beneficial especially in the settings where there are nonlinear relationships between covariates and rewards and when the covariates have high-dimensionality (with some sparsity assumptions). 

We show through simulations that our algortihms outperform the state-of-the-art methods in highly nonlinear settings. 

## Notes
Work in progress.
Co-Authors: Jamie Kang(jamiekang@stanford.edu) and Lin Fan (linfan@stanford.edu)

## References
* H. Bastani and M. Bayati. Online Decision-Making with High-Dimensional Covariates. 2015 https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2661896
* A. Goldenshluger and A. Zeevi. A Linear Response Bandit Problem. Stochastic Systems Vol 3. 2013
