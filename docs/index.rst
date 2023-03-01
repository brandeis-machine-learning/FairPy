.. figure:: fairpy_logo.png
    :scale: 6%
    :alt: logo

----

FairPy Documentation
=====================

FairPy is a comprehensive **Python Library** for **Machine Learning Fairness**, covering various fairness notions, data structures, and learning tasks. 
FairPy is a good toolkit for mitigating bias and helping to deliver equitable outcomes from machine learning models.

FairPy includes many latest fairness algorithms, such as LabelBias :cite:p:`jiang2020identifying` and FairGLM :cite:p:`do2022fair`.
For consistency and accessibility, FairPy is developed on top of `scikit-learn <https://scikit-learn.org/>`_ and `PyTorch <https://pytorch.org/>`_.

----

Why FairPy: Compare to Other Libraries
=======================================

=======================  ==========  =========  ===============  ==========
Attribute                FairPy      Fairlearn  AI Fairness 360  inFairness
=======================  ==========  =========  ===============  ==========
Group Fairness           [x]         [x]        [x]              [ ]       
Individual Fairness      [x]         [ ]        [x]              [x]    
Minimax Fairness         [x]         [ ]        [ ]              [ ]       
Tabular Data             [x]         [x]        [x]              [x]    
Graph Data               [x]         [ ]        [ ]              [ ]       
Classification           [x]         [x]        [x]              [x]    
Regression               [x]         [ ]        [x]              [ ]       
Ranking                  [x]         [ ]        [ ]              [ ]       
Number of Algorithms     10+         4          14               3
Compatible with sklearn  [x]         [x]        Partially        [ ]       
Latest release           March 2023  July 2021  March 2021       June 2022
=======================  ==========  =========  ===============  ==========

----

Sample Code of FairPy
======================


.. code-block:: python

    from fairpy.dataset import Adult
    from fairpy.model import LabelBias

    dataset = Adult()                  # Choose or customize a dataset
    split_data = dataset.split()       # Extract features, labels, and sensitive attributes from the dataset
    model = LabelBias()                # Choose a fairness algorithm
    model.fit(split_data.X_train, split_data.y_train, split_data.s_train)  # Fit model with provided data
    model.predict(split_data.X_test)   # Predict

----


Implemented Algorithms
=======================

FairPy toolkit consists of three major functional groups:

**(i) Fairness Algorithms** :

====  ==============  ==============  ====  ====
Type  Name in FairPy  Task            Year  Ref
====  ==============  ==============  ====  ====
Pre   LabelBias       Classification  2020  :cite:p:`jiang2020identifying`
Pre   LinearFairERM   Any             2018  :cite:p:`donini2018empirical`
Pre   DIRemover       Any             2015  :cite:p:`feldman2015certifying`
Pre   IFair           Any             2019  :cite:p:`lahoti2019ifair`
In    FairCstr        Classification  2017  :cite:p:`zafar2017fairness`
In    FairGLM         Classification  2022  :cite:p:`do2022fair`
In    FairPGRank      Ranking         2019  :cite:p:`singh2019policy`
Post  EqOddsCalib     Classification  2016  :cite:p:`hardt2016equality`
Post  FairRank        Ranking         2017  :cite:p:`zehlike2017fa`
====  ==============  ==============  ====  ====

**(ii) Datasets** :

=====  ==============
Type   Name in FairPy
=====  ==============
Table  Adult
Table  Bank
Table  Compas
Table  Credit
Table  Dutch
Table  German
Table  Oulad
Table  Xing
=====  ==============


**(iii) Metrics** :

==============  ==============  
Type            Name in FairPy
==============  ==============
Classification  binary_dp
Classification  binary_eop
Ranking         xAUC
Ranking         dcg
==============  ==============


----

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   install

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API

   fairpy.dataset
   fairpy.metric
   fairpy.model

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   reference