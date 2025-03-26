# When mitigating bias is unfair: multiplicity and arbitrariness in algorithmic group fairness

This repository contains the codebase for "When mitigating bias is unfair: multiplicity and arbitrariness in algorithmic group fairness" (on [Arxiv](https://arxiv.org/abs/2302.07185)), published in IEEE SaTML 2025.

Most research on fair machine learning has prioritized optimizing criteria such as Demographic Parity and Equalized Odds. Despite these efforts, there remains a limited understanding of how different bias mitigation strategies affect individual predictions and whether they introduce arbitrariness into the debiasing process. This paper addresses these gaps by exploring whether models that achieve comparable fairness and accuracy metrics impact the same individuals and mitigate bias in a consistent manner. We introduce the FRAME (FaiRness Arbitrariness and Multiplicity Evaluation) framework, which evaluates bias mitigation through five dimensions: Impact Size (how many people were affected), Change Direction (positive versus negative changes), Decision Rates (impact on models' acceptance rates), Affected Subpopulations (who was affected), and Neglected Subpopulations (where unfairness persists). This framework is intended to help practitioners understand the impacts of debiasing processes and make better-informed decisions regarding model selection. Applying FRAME to various bias mitigation approaches across key datasets allows us to exhibit significant differences in the behaviors of debiasing methods. These findings highlight the limitations of current fairness criteria and the inherent arbitrariness in the debiasing process.


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
```

## Data
The following datasets are used for the analysis:
* [Adult Census](https://archive.ics.uci.edu/ml/datasets/adult)
* [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank%2BMarketing)
* [Credit Card Default](https://archive.ics.uci.edu/ml/datasets/default%2Bof%2Bcredit%2Bcard%2Bclients)
* [Dutch Census](https://microdata.worldbank.org/index.php/catalog/2102/data-dictionary)
* [COMPAS](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)
