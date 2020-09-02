# Network_Intrusion_Detector
Problem Statement

Train both a Convolutional Neural Network and a Fully Connected Dense Neural Network to detect an attack on a network based on the network activity.

METHODOLOGY
Considering the problem

	We were provided a dataset of >490k records. The records in actuality are a 10% subset of a larger dataset. Considering the problem I decided to systematically wrangle the data to create an unbiased dataset for training.

Removing unneeded data
I first removed the duplicates in the dataset using the function “.drop_duplicates()”
This substantially reduced the dataset from <494k records to >150k records
Balanced the records
There were more “normal” records than records that indicated an “attack”. To resolve this I implored the following steps.
Separated the data into two data frames
One containing only attacks and the other containing only normal
Shuffle the normal data
Data may have some order so removing random records reduces the risk of introducing a bias to the model
Reduce the normal data so that the number or normal records matches the number of attack records
Recombine the now equalized data into a single dataframe
Finally again shuffle the data. 
This further reduces the risk of introducing a bias to the model
Encoding discrete values
Several of the columns are for discrete values that indicate a type. These records need to be encoded so the model can identify them.
Separate columns of discrete values into their own individual dataframe’s and remove those columns from the previously reduced dataset.
Protocol_type
Service
Flags
Outcomes
One Hot Encoded all discrete values
Re append the one hot encoded columns back to the dataframe. (with the acception of the Outcomes column)
This column needs more processing
Managing Outcomes column
Because there are 22 different identified attacks in this dataset we need to first train the model to identify the difference between attack and normal traffic.
All 22 attacks fall into a single attack category
To simplify my efforts I One Hot encoded the Outcome column.
Extracted the column that represented normal activity
This column has a ‘1’ for all normal activity and ‘0’ for anything that is an attack.
Per the assignment instructions we need normal activity to be represented with a ‘0’ and attacks represented with a ‘1’
I inverted the values of the extracted ‘normal’ column using the function parameters. “df.replace({1:0,0:1})”
I now have my ‘y’
Normalize the x_dataframe
The data contained in the x_dataframe contains only numbers however many of the cells contain large numbers which will greatly reduce the training speed.
Steps for normalizing the data
Pass dataframe.values into a new variable “values = dataframe.values”
Create min_max_scaler using “scaler = preprocessing.MinMaxScaler()”
Then fit_transform the values using the scaler “scaler.fit_transform(values)”
Convert the scaled values back into a dataframe
X_dataframe is now ready for “train_test_split()”

	At multiple points throughout this process I felt it necessary to save the converted data. This part in particular because I could read in the data from anywhere and train a model without needing to process the data. So I created a new dataframe with the appended y column and saved the data as a csv file. 


Splitting data for CNN
Because a CNN requires the data to be in a 4 dimensional shape and a Fully Connected Neural Network requires the data to be in a 2 dimensional shape I felt that the best strategy was to 2 versions of test train. One where the x, y, split is converted into 4D and the other where the data is converted into 2D
CNN data
For x train/test i used the functions... “.to_numpy().reshape((x_network_train.shape[0], 1, x_network.shape[1], 1))”
This allowed me to not have to change the values if I later used the entire dataset and not the subset for training.
For y train/test i used the functions… “tf.keras.utils.to_categorical(y_network_train.to_numpy(), 2)”
This again was done with the intention of later using the whole dataset for training
CNN model
After some initial confusion regarding the batch_size I was able to build a very good model.
I created a sequential model 
Batch_size = 128
Using conv2D with 32 filters, kernel size (1,2), strides(1,1), padding ‘same’, activation ‘relu’ input_shape (x_train_cnn.shape[1:]) 
Again i used the dataframe’s shape to that in the future I would not need to change the values in the model when I used the entire dataset…
I added several more layers including MaxPooling(size(1,2))
Another convolutional layer with 64 filters, all other parameters were the same as the first convolution
Another MaxPooling, then Flatten then two dense layers 128
The output layer had 2 outputs and used softmax as its activation
I compiled the model using “categorical_crossentropy” with adam as the optimizer
I only needed to run through 10 epochs and the model came out very well.
To be honest the results were so good that I thought there must be something wrong. Since it was 99.8% accurate.
After rechecking my data and the model for possible over fitting then speaking with multiple students and the professor I conceded that my data was correct.
Here are the metrics for CNN




Here are the metrics for the DENSE Neural Network







Task Division
Shane Harris (Team Leader)
Read in data
Proposed chain of events to eliminate bias
Removed Duplicates of data
One Hot Encoded all discrene columns
Prepare data for training
Separated x for y
Created 2 versions of train_test_split
Build CNN Model
Tuned CNN model
Build Dense Model
Tuned Dense Model
Retuned Dense Model
Build Regression Model because Dense model would not train
Started writeup


Adrian Cabreros
Equalized normal records and attack records
Converted data to categorical
Trained dense model with categorical data
Created matplot lib matrix and results for dense model
Michale Dorst
Converted regression model into categorical
Research and debugging
Pair programming (navigator)

Challenges
	This project started out easy enough. Preparing the data did not take me long even with my limited python experience. The real issues started with the models.
I had built the CNN model and everything looked like it should work. Yet the model would not progress passed the first epoch. This confounded me for several days until I finally went to my professor who eventually pointed out that my batch size was 1 and needed to be much larger. After this the CNN model came along very well.
	I then started on the dense model. This model would not learn. I spent 5 days straight working on this model. Nothing I tried would work. I was not until another team member took a look at the code with fresh eyes and was able to find a solution. He turned the y variable to categorical and finally the model began to learn. We ran out of time to train our models on the individual attack types. However; I plan to continue on this project in my free time because I found this project fascinating.
