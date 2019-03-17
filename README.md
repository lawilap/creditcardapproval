Predicting Credit Card Approvals using Logistic Regression
Banks can’t approve all the requests for credit cards. The banks look at several factors to determine the creditworthiness of a person. Their ideology is to determine if a person can pay his or her bills on time.

•	The data used is from UCI’s Machine learning repository: http://archive.ics.uci.edu/ml/datasets/credit+approval

•	It consists of 15 variables which represents various attributes of the customer. These variables are anonymised to protect the privacy of the customers. The 16th variable represents whether the banks approve or reject the application for that individual in the form of a + or –

•	The data also has some missing values. Some model may not work on missing data. There may be some important information which can be used to train the model.

Imputed data as follows to improve accuracy of the model:

•	We have some data as ‘?’, we replace them with NaN using numpy

•	We impute the missing numeric data with mean values for that variable.

•	We impute the missing non-numerical data with the most frequent values for that variable.

Data Pre-processing: 

•	LabelEncoder normalizes the data and It can also be used to transform non-numerical data to numerical labels, if they are comparable.

•	Label encoding is an important task for this project as we need to find correlation between variables and then run prediction algorithms on the data.

•	We check the correlation and observe feature 4,3 and 9,10 are highly correlated, this might lead to multicollinearity

•	We can also observe that feature 0,6,11 have very less correlation with variable 15 which is our dependent variable. So, we decide to drop features 0,4,6,9,11


•	Next we Split the data into training and test sets into 70/30 ratio, which means the 70% of the data will be used for training our model and we will test our model on the remaining 30% data.

•	we build a confusion matrix to see how well our model has performed. The confusion matrix is a specific table layout that allows visualization of the performance of an algorithm. It gives the information about the accuracy measures in a predictive model. Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class. All correct predictions are in the diagonal of the table i.e. true positives and true negatives.

•	We then create KNN model with K=5 and check the accuracy

•	Next we build our second model using Logistic Regression and check accuracy

•	We observe that accuracy has improved significantly

•	We plot the ROC curve to tune our model and increase the accuracy

•	We change the threshold value for logistic regression to 0.4 (which is 0.5 by default). This means that any probability below 0.4 will be considered 0 and anything above this will be considered 1.

•	We observe a significant increase in Accuracy and Precision rate after changing the threshold.
