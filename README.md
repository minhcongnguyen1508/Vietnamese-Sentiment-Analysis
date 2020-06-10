# Vietnamese-Sentiment-Analysis
5 class &amp; 3 labels POS, NEU, NEG

How to run models

# Use Visual Studio Code
Step 1: Clone git 

Step 2: Install Package in file requirement.txt & Activate environment:

        cd Vietnamese-Sentiment-Analysis
        
        pip3 install -r requirement.txt
        
     or pip install -r requirement.txt

Step 3: cd src

        Run file demo.py

# If you want train new models. 

Step 1: Create new domain Word.model. Change code in file train.py
        
        Uncomment:
        
        reviews, y_labels = load_data_from_dir('../data/raw')
        
        y_5class_labels, dis = y2labels(y_labels)

Step 2: Add new data train & valid path in file train.py ("# load data")

Step 3: Train model "python train.py"

Accuracy for 3 labels POS, NEU, NEG is 80 %

Accuracy for 5 lable is 60%

Use Convolution Neural Network

@Author: Minh Cong
