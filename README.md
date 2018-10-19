<img src="https://github.com/jarty13/Wine-Rating-impact-on-Wine-Price/blob/master/wine.png" width="250" height="350">

# Project Motivation
* Is there a linear relationship between the price of wine and the wine rating? If the relationship is linear, is there an opportunity to increase the price of the wine to drive more revenue?
* Can wine reviews/descriptions predict the price category of the of the wine?

# Data Clean-up and Exploration
* Utilizing the Wine Enthusiast dataset from Kaggle, we analyzed the wine reviews/descriptions to see if it had an impact on the price of wine that was produced in the US, Europe, Australia, and Argentina
* Data included over 130K wine reviews from wines produced in the US, Europe, Australia, and Argentina 
* Implemented Logistic Regression Classifier  to see if the wine description could accurately categorize if the wine was categorized as "Cheap"( <$15), "Low"( $15- $40)," Medium"($40-$100), "High"(>$100)
* Reduced dimensionality of the data by categorizing the wines into 19 different categories ranging from dry/sweet reds to dry/sweet white wines.
* Vectorized wine descriptions using Scikit-learn, TfidfVectorizer to get a unique list of all words mentioned in the wine reviews

<img src="https://github.com/jarty13/Wine-Rating-impact-on-Wine-Price/blob/master/images/wine%20rating.png" width="650" height="450">

<img src="https://github.com/jarty13/Wine-Rating-impact-on-Wine-Price/blob/master/images/wine%20price%20distribution.png" width="650" height="450">

<img src="https://github.com/jarty13/Wine-Rating-impact-on-Wine-Price/blob/master/images/price%20distribtuion%20by%20type%20of%20wine-%20US.png" width="850" height="450">

<img src="https://github.com/jarty13/Wine-Rating-impact-on-Wine-Price/blob/master/images/price%20distribution%20by%20type%20of%20wine%20-%20other%20countries.png" width="850" height="450">

# Results:
* In regards to the relationship to price and rating, we saw that there was a small but linear relationship in the price of wine produced in the US, resulting in   R-Square of 66%. While the Wines in Europe, Argentina, and Australia had no linear relationship since they had higher priced wine ranging above $250 that were reviewing ratings at our mean rate per bottle of wine or 88.
*  Running a logistic regression to predict the price of wine off of the words in a wine review resulted in an overall accuracy if 66%.
* To improve accuracy will be to run a grid search, categorize the wines into more defined wine priced categories and running additional classifiers on our train and test data.
* There is an opportunity to increase prices on certain wines if their reviews are over 88pts with the price point in the low range, in order to increase revenue. 
