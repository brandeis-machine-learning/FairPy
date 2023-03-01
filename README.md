<p align="center">
    <img width="450" src="https://raw.githubusercontent.com/brandeis-machine-learning/FairPy/dev/docs/fairpy_logo.png" alt="fairpy logo">
</p>
<hr>

**FairPy** is a comprehensive python library for **Machine Learning Fairness**, covering various fairness **notions**, multiple advanced fair **algorithms**, and corresponding experimental **datasets**.
FairPy served as a good toolkit for mitigating bias and helping to deliver equitable outcomes from machine learning models.

Reach Fairness with few lines of codes:
```python
from fairpy.dataset import Adult
from fairpy.model import LabelBias

dataset = Adult()
split_data = dataset.split()
model = LabelBias()
model.fit(split_data.X_train, split_data.y_train, split_data.s_train)
model.predict(split_data.X_test)
```

Interested in machine learning fairness? Check our actively maintained paper and resource list: [Awesome Machine Learning Fairness](https://github.com/brandeis-machine-learning/awesome-ml-fairness).

## Installation

To install FairPy via **pip**, run the command:

    pip install fairpy

### Optional Dependencies

There are some optional dependencies beyond a regular installation. These dependencies are only used for some specific algorithms. 
You can install them manually in need and fit them to your local environment.

 - [PyTorch](https://pytorch.org/): used for all **deep learning models**
 - [gurobipy](https://www.gurobi.com/documentation/9.5/quickstart_linux/cs_using_pip_to_install_gr.html): commercial optimization software, used for **InflFair**

## Algorithms

We implement the following algorithms in FairPy. Algorithms are divided into three types: **pre-processing**, **in-processing**, and **post-processing**.

| Type | Name in FairPy | Task           | Year | Ref. |
|:----:|:--------------:|:--------------:|:----:|:----:|
| Pre  | LabelBias      | Classification | 2019 | [2]  |
| Pre  | LinearFairERM  | Any            | 2018 | [3]  |
| Pre  | DIRemover      | Any            | 2015 | [4]  |
| Pre  | IFair          | Any            | 2019 | [10] |
| Pre  | InflFair       | Classification | 2022 | [9]  |
| In   | FairCstr       | Classification | 2017 | [5]  |
| In   | FairGLM        | Classification | 2022 | [7]  |
| In   | FairPGRank     | Ranking        | 2019 | [12] |
| Post | EqOddsCalib    | Classification | 2016 | [6]  |
| Post | FairRank       | Ranking        | 2018 | [11] |

## Datasets

We collect the following datasets for easy experiments. Data will be automatically downloaded in FairPy.

1. Adult Data Set
1. COMPAS Recidivism Risk Score Data and Analysis
1. Bank Marketing Data Set
1. Statlog (German Credit Data) Data Set
1. default of credit card clients Data Set
1. The Dutch Virtual Census of 2001 - IPUMS Subset
1. Open University Learning Analytics dataset
1. Xing Dataset

## Reference

1. Data preprocessing techniques for classification without discrimination  
2. Identifying and Correcting Label Bias in Machine Learning  
3. Empirical risk minimization under fairness constraints  
4. Certifying and Removing Disparate Impact  
5. Fairness Constraints: Mechanisms for Fair Classification  
6. Equality of opportunity in supervised learning  
7. Fair Generalized Linear Models with a Convex Penalty  
8. The Fairness of Risk Scores Beyond Classification: Bipartite Ranking and the xAUC Metric  
9. Achieving Fairness at No Utility Cost via Data Reweighing with Influence  
10. iFair: Learning Individually Fair Data Representations for Algorithmic Decision Making  
11. FA*IR: A Fair Top-k Ranking Algorithm  
12. Policy Learning for Fairness in Ranking  



