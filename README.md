helicopter
==============================

Install
-------

    make requirements

Data
----

Input data should be inside `data/raw` directory. Type `make data`
to generate data.
    
Data files are:

- Data_Main_rotor.csv
- Data_Tail_rotor.csv
- manoeuver_code.csv
- MCA_code.csv
- PCM.csv

Experiments
-----------

Experiment scripts `train_model.py` (for mdn) and `train_siamese.py` (for siamese networks)
 are available in the `src/models` directory. 

Figures displaying the loss are automatically generated in the tensorboard log
directory. They are generated automatically when training models. 
Use `tensorboard --logdir {path to log dir}` in order to see
them in your web browser.

Figure displaying the performance of mdn are generated automatically after
execution of `train_model.py`.
