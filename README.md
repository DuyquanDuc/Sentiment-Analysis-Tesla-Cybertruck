# About Sentimental-Analysis: Tesla Upcoming Cybertruck Project

The unicorn product of Tesla, Cybertruck, is without doubt one of the most eccentric-designed, versatile one among its own kind. However, there is not a small number of people find its design to be unaesthetic, and come with unneccessary additions. This sentiment-analysis is aiming to find out how welcoming the public are to the Cybertruck, and what kinds of criticisms are the most popular right now. So that, we could assess the impact the Cybertruck will make on the industry, and how much revenue will it generate for the company.

## Collaborating 

-In order for this project to success, not only data analysis process has to be precise and efficient, we will also need in-depth knowledge on the truck market. So what can collaborators do to contribute to this project ?

1. Read and aggregate articles on Cybertruck：Put it in 1 file what you consider most impactful point to the analysis.
- NGUYEN HA GIA HUY
2. Cleaning and simple wrangling.
- NUYEN MINH HIEU

## Analysis Conducting Procedure

### 1. Data Collecting 

-This maybe the most cumbersome part of this projects as many questions need to answered in order to have proper, high-influential datasets such as:

+ What kind of data will help us understand the current sentiment toward Cybertruck ? - Tweets, Facebook Posts, Reddit posts... these are for sentiment analysis, we want to go a step further to take data from Tesla on the number of pre-ordered truck, and how its market will affect the sales.
+ Where will we get the data from ? Twitter, Facebook, Reddit, and Tesla.
+ How will we get it ? Through access to APIs of those SMS, and scraping Tesla website data.

### 2. Data Wrangling
- This part includes cleaning, standardizing and feature engineering (if neccessary): 
+ Using Dataiku DSS, Pandas and Numpy for cleaning and features transformation
+ spaCy for tokenizing, lemmatizing, stopwords deleting
### 3. Preparing data for training
- This part should be the same as step 2, but since its original, we need to label manually, or finding out pre-trained model to label the data before train: 
+ After researching I decided to use Google Languagev1 in the package google.api to classify the dataset. It yielded results range from -1 to 1 with -1 as most negative, 1 as most positive
### 4. Modelling
-  We are trying out Logistic Regression using sklearn and numpy: 
+ Using simple logistic regression from spaCy. Accuracy yielded approximately 0.8.
### 5. Visualizing
- Expected result is to show different sentiment toward diffent aspects of the Cybertruck. Thus estimate which one should be made changes to:
+ Built a sentiment chart around a selected topic. In this case I choose FSD (Full-self-driving) feature.

#Conclusion

After successfully conducting this analysis, these code can be reused and the project can be expanded substantially depending on the person's purposes. Special thanks to NGUYEN THANH HAI and SAM CRAWSHAYS for supportting me with this project. Others CREDITS goes for NGUYEN HUNG HAO for collectting the twitter data, NGUYEN MINH HIEU for cleaning, and NGUYEN HA GIA HUY for market researching. Honorable mentions team GO Digital !  
