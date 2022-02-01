# Grocery-sales-prediction
## Problem statement
To predict the sales of BIG MART GROCERY DATA test set, we have a training set with input features (different item properties and location of stores)\
**Data reference:** https://www.kaggle.com/brijbhushannanda1979/bigmart-sales-data 
## Approach
We will analyse the data given, preprocessing and look into performances of various ML algorithms in prediction of BIG MART grocery sales.\
Observe performances of different ML algorithms 
- 'Linear Regression'
- 'Ridge Regression'
- 'Lasso Regression'
- 'Elastic Net Regression'
- 'SGD Regression'
- 'SVM'
- 'Decision Tree'
- 'Random Forest'
- 'Ada Boost'
- 'Bagging'
- 'Gradient Boost'
## Observation
We found out Gradient Boosting algorithm perform better with RMSE score of 1104.74 among all other we observed. 

## Project outline
Extract data into pandas from drive 

```bash
 df=pd.read_csv("/content/drive/MyDrive/ML projects/Grocery sales prediction/Train.csv")
```

After observing the statistical information and null values, we initially find out which are actually important features\
Then comes data preprocessing, convert object values into categorical values. Removing null values and reducing complex data into simpler form. 

```bash
 fig,axes=plt.subplots(1,1,figsize=(10,8))
 sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=df)
 plt.plot([69,69],[0,5000])
 plt.plot([137,137],[0,5000])
 plt.plot([203,203],[0,9000])
```
![Screenshot (2259)](https://user-images.githubusercontent.com/65950195/151908448-eb0eddb6-e9c2-4d13-97a4-a00ebb6559ea.png)

