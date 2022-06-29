# Contextual Bandit Experiment
Section 6.2 of the paper

We compare with [Majzoubi et. al.](https://arxiv.org/abs/2006.06040) on their [collection of OpenML regression problems](https://github.com/VowpalWabbit/vowpal_wabbit/tree/master/papers/cats/code_submission).

1. The data can be obtained from here:
  https://www.openml.org/data/get_csv/150677/BNG_wisconsin.arff       --> BNG_wisconsin.csv
  https://www.openml.org/data/get_csv/150680/BNG_cpu_act.arff         --> BNG_cpu_act.csv
  https://www.openml.org/data/get_csv/150679/BNG_auto_price.arff      --> BNG_auto_price.csv
  https://www.openml.org/data/get_csv/21230845/file639340bd9ca9.arff  --> black_friday.csv
  https://www.openml.org/data/get_csv/5698591/file62a9329beed2.arff   --> zurich.csv
1. and then processed like so:
```console
foo@bar:~$ wget https://www.openml.org/data/get_csv/150677/BNG_wisconsin.arff -O BNG_wisconsin.csv
foo@bar:~$ sed '/^$/d' -i BNG_wisconsin.csv
foo@bar:~$ python ./preprocess_data.py --csv_file BNG_wisconsin.csv 
```
1. [This interactive notebook](fastcbcorral.ipynb) demonstrates the technique, and also contains the results in the table in the paper.
    1. As per Majzoubi et. al., confidence intervals are computed from a single online run, with hyperparameters chosen to be optimal in hindsight.
    1. [Hyperparameter optimization](tune-fastcbcorral.py) is by random search, e.g., 
```console
foo@bar:~$ python ./tune-fastcbcorral.py black_friday.dat > tune-fastcbcorral.black_friday.res
```
and then process the output to determine the best hyperparameter settings.
