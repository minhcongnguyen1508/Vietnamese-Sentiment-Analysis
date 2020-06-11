# Vietnamese-Sentiment-Analysis
5 class &amp; 3 labels POS, NEU, NEG

How to run models

# Install
Step 1: clone git https://github.com/minhcongnguyen1508/Vietnamese-Sentiment-Analysis.git

Step 2: Activate environment (if you use venv) & install package in file requirement.txt:

        cd Vietnamese-Sentiment-Analysis
        

Step 3: cd Vietnamese-Sentiment-Analysis/src

        Run file demo.py

# If you want train new models. 

Step 1: Create new domain Word.model. Change code in file train.py
        
        Uncomment:
        
        reviews, y_labels = load_data_from_dir('../data/raw')
        
        y_5class_labels, dis = y2labels(y_labels)

Step 2: Add new data train & valid path in file train.py ("# load data")

Step 3: Train model "python train.py"

# Result:

Accuracy (3 labels POS, NEU, NEG): 80 %

Accuracy (5 lable): 60%

# Techniques:

Using Convolution Neural Network

# Improvement:
Now, My model is biased for Label 5. => Model quality is very low. You can see confusion matrix.

You can improve my model based on Oversampling and data Augmentation techniques.


@Author: Minh Cong
