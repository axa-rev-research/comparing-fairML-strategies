# When Mitigating Bias is Unfair: A Comprehensive Study on the Impact of Bias Mitigation Algorithms

This repository contains the codebase for "When Mitigating Bias is Unfair: A Comprehensive Study on the Impact of Bias Mitigation Algorithms" (on [Arxiv](https://arxiv.org/abs/2302.07185)).

Most works on the fairness of machine learning systems focus on the blind optimization of common fairness metrics, such as Demographic Parity and Equalized Odds. In this paper, we conduct a comparative study of several bias mitigation approaches to investigate their behaviors at a fine grain, the prediction level. Our objective is to characterize the differences between fair models obtained with different approaches. With comparable performances in fairness and accuracy, are the different bias mitigation approaches impacting a similar number of individuals? Do they mitigate bias in a similar way? Do they affect the same individuals when debiasing a model? Our findings show that bias mitigation approaches differ a lot in their strategies, both in the number of impacted individuals and the populations targeted. More surprisingly, we show these results even apply for several runs of the same mitigation approach. These findings raise questions about the limitations of the current group fairness metrics, as well as the arbitrariness, hence unfairness, of the whole debiasing process.

## Structure

```
.
├── README.md
├── fairlearn_int <-- [Fairlearn](https://fairlearn.org/) with slight modifications to allow working with multiple different data structures
├── fairness 
|   ├── helpers <-- file containing helper functions
|   ├── avd_helpers  <-- file containing helper functions
├── notebooks <-- folder containing experiments for each analysed dataset, structured in the following way:
|   ├── name_dataset
|   |   ├── name
|   |   ├── runs
├── results <-- folder containing experiment results necessary for further analyses for each analysed dataset, structured in the following way:
|   ├── name_dataset
|   |   ├── results

## Data
The following datasets are used for the analysis:
* [Adult Census](https://archive.ics.uci.edu/ml/datasets/adult)
* [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank%2BMarketing)
* [Credit Card Default](https://archive.ics.uci.edu/ml/datasets/default%2Bof%2Bcredit%2Bcard%2Bclients)
* [Dutch Census](https://microdata.worldbank.org/index.php/catalog/2102/data-dictionary)
* [COMPAS](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)
