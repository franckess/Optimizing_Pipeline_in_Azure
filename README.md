<img align="left" width="100" height="75" src="https://github.com/dpbac/Optimizing-an-ML-Pipeline-in-Azure/blob/master/images/microsoft-azure-640x401.png">

# Optimizing an ML Pipeline in Azure


## Overview

This project is part of the Udacity Azure ML Nanodegree.

For this project, our main tasks were to build and optimize an Azure ML pipeline using [Scikit-learn Logistic Regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). 
The hyperparameters of this model are optimized using **HyperDrive**. This model is then compared to **Azure AutoML** so that the results obtained by both models are compared.

A diagram illustrating the steps of this project is shown below:

![](https://github.com/dpbac/Optimizing-an-ML-Pipeline-in-Azure/blob/master/images/project_summary.JPG)
source: Nanodegree Program Machine Learning Engineer with Microsoft Azure

## Summary

### Problem Statement

This dataset contains information related with direct marketing campaigns (via phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe (_yes/no_) a term deposit. 

**Input variables:**

1. `age`: (numeric)
2. `job`: type of job (categorical)
3. `marital`: marital status (categorical)
4. `education`: (categorical)
5. `default`: has credit in default? (categorical)
6. `housing`: has housing loan? (categorical: 'no','yes','unknown')
7. `loan`: has personal loan? (categorical)
8. `contact`: contact communication type (categorical)
9. `month`: last contact month of year (categorical)
10. `day_of_week`: last contact day of the week (categorical)
11. `duration`: last contact duration, in seconds (numeric)
12. `campaign`: number of contacts performed during this campaign and for this client (numeric)
13. `pdays`: number of days that passed by after the client was last contacted from a previous campaign (numeric)
14. `previous`: number of contacts performed before this campaign and for this client (numeric)
15. `poutcome`: outcome of the previous marketing campaign (categorical)
16. `emp.var.rate`: employment variation rate - quarterly indicator (numeric)
17. `cons.price.idx`: consumer price index - monthly indicator (numeric)
18. `cons.conf.idx`: consumer confidence index - monthly indicator (numeric)
19. `euribor3m`: euribor 3 month rate - daily indicator (numeric)
20. `nr.employed`: number of employees - quarterly indicator (numeric)

**Output variable (desired target):**

21. `y` - has the client subscribed a term deposit? (binary: _'yes','no'_)

**original source of the data**: 
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, 
June 2014 (https://repositorio.iscte-iul.pt/bitstream/10071/9499/5/dss_v3.pdf )

### Solution & Results 

As mentioned above, we proceeded with the two following approaches:

1. Scikit-learn Logistic Regression model: where two hyperparameters were optimized using **HyperDrive**.
2. Azure AutoML: which allowed AzureML to go through different models and come up with a single model which best optimized the metric of interest (i.e. `accuracy`).

Using **HyperDrive**, we were able to obtain an accuracy of **90.74%**. Using the **AutoML**, and the best performing model was `Voting Ensemble`, which achieved an accuracy of **91.67%**.
## Scikit-learn Pipeline

### Summary of the Pipeline

The script `train.py` included a few strategic steps:

1. Loading dataset.
2. Cleaning and transforming data (e.g. drop NaN values, one hot encode, etc.). 
3. Calling the SKlearn Logistic Regression model using parameters:
    * `--C` (float): Inverse of regularization strength
    * `max_iter`(int): Maximum number of iterations taken for the solvers to converge

The following steps were run from the main notebook:

1. Initialize our `Workspace`
2. Create an `Experiment`
3. Define resources, i.e., create `AmlCompute` as training compute resource
4. `Hyperparameter tuning` i.e. defining parameters to be used by **HyperDrive**, which also involved specifying a `parameter sampler`, a `policy` for early termination, and creating an estimator for the `train.py` script.
5. Submission the `HyperDriveConfig` to run the experiment.
6. Use ` get_best_run_by_primary_metric()` on the run to select the best combination of hyperparameters for the Sklearn Logistic Regression model
7. Save the best model.

### What are the benefits of the parameter sampler chosen?

In the `random sampling` algorithm used in this project, parameter values are chosen from a set of discrete values (`choice`) or randomly selected over a uniform distribution

The other two available techniques (Grid Sampling and Bayesian) are indicated if you have a budget to exhaustively search over the search space. In addition, Bayesian does not allow using early termination.

### What are the benefits of the early stopping policy you chosen?

`Early stopping policy` automatically terminates poorly performing runs.

The `early termination policy` we used [`Bandit Policy`]( https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?preserve-view=true&view=azure-ml-py#&preserve-view=truedefinition ). This policy is based on `slack factor/slack amount` and `evaluation interval`. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount. This allows more aggressive savings than Median Stopping policy if we apply a smaller allowable slack.

Parameter `slack_factor` which is the slack allowed with respect to the best performing training run, need to be defined while `evaluation_interval` and `delay_interval` are optional.

## AutoML

**AutoML** tries different models and algorithms during the automation and tuning process within a short period of time. The best performing model was `Voting Ensemble` with an accuracy of **91.67%**.
## Pipeline comparison

Although the performance of **AutoML** (`Voting Ensemble`) was slightly better than HyperDrive, it didn't demonstrate a significant improvement (less than 2%).

**AutoML** is definitely better than **HyperDrive** in terms of architecture since we can create hundreds of models a day, get better model accuracy and deploy models faster.
## Future work

The first point to consider that the data is **highly imbalanced** (88.80% is labeled NO and 11.20% is labeled YES). This imbalance can be handled by using technique like **Synthetic Minority Oversampling Technique** (a.k.a. **SMOTE**) during the data preparation step.

We could include additional hyperparameters used in Sklearn Logistic Regression in order to achieve better results in the future. Using different parameter sampling techniques and tuning the arguments of the BanditPolicy can also prove fruitful.

About the AutoML, we would like to tune more config parameters; increasing experiment timeout minutes will enable us to test more models and thus improving the performance.

## Proof of cluster clean up

We ran the following command:
```python
aml_compute.delete()
```

![](https://drive.google.com/file/d/1QuiSW6UIArEwy5XdD8Egz75BLBn-s0Bi/view?usp=sharing)


