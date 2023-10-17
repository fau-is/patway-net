# Interpretable recurrent deep neural networks: The case of patient pathway prediction

#Setup of PatWay-Net
   1. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) 
   2. After setting up miniconda you can make use of the `conda` command in your command line (e.g. CMD or Bash)
   3. To install required packages run `pip install -r requirements.txt` inside the root directory.
   4. Train and test the PatWay-Net (pwn) Model by executing `main.py`. You can select the used dataset for your run in the `if __name__ == "__main__":`-Section of the script.
   5. Create plots about statistics of the pwn model you trained by executing any of the scripts with the prefix `interpret_` or with the suffix `plot`.

# Data Sets
* Sepsis (https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639)
* BPI2012 (https://www.win.tue.nl/bpi/doku.php?id=2012:challenge)
* Hospital Billing (https://research.tue.nl/en/datasets/hospital-billing-event-log)

For BPI2012 and Hospital Billing, we use the preprocessed versions, namely "bpic2012_O_ACCEPTED-COMPLETE" and "hospital_billing_2.csv", shared by [Teinemaa et al. (2018)](https://github.com/irhete/predictive-monitoring-benchmark).   

# Hyperparameter Settings for BPI2012 (for seed 15)
![Hyperparameters](bpi2012.JPG?raw=true "Hyperparameter settings")

# Hyperparameter Settings for Hospital Billing (for seed 15)
![Hyperparameters](hospital_billing.JPG?raw=true "Hyperparameter settings")
