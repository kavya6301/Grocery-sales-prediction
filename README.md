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
 #Plotting and Binning the data
 fig,axes=plt.subplots(1,1,figsize=(10,8))
 sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=df)
 plt.plot([69,69],[0,5000])
 plt.plot([137,137],[0,5000])
 plt.plot([203,203],[0,9000])
```
![image](https://user-images.githubusercontent.com/65950195/151908744-d29484ad-0e49-4fba-abd3-7fda77b0f429.png)

```bash
 data[data.Outlet_Size.isnull()]
```
![image](https://user-images.githubusercontent.com/65950195/151909067-ca815b7c-ac01-4d28-9f3d-e1c9a8a49c31.png)

```bash
#Changing null values according to the observation
 def func(x):
    if x.Outlet_Identifier == 'OUT010' :
        x.Outlet_Size == 'High'
    elif x.Outlet_Identifier == 'OUT045' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT017' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT013' :
        x.Outlet_Size == 'High'
    elif x.Outlet_Identifier == 'OUT046' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT035' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT019' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT027' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT049' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT018' :
        x.Outlet_Size == 'Medium'
    return(x)
```
Now our data is good. By applying different ML algorithms, we get RMSE scores as follows

```bash
 go=pd.DataFrame({'RMSE':[lr_score,lr_score_cross,r_score,r_score_cross,l_score,l_score_cross,en_score,en_score_cross,
                     sgd_score,sgd_score_cross,svm_score,svm_score_cross,dtr_score,dtr_score_cross,rf_score,rf_score_cross,
                     ada_score,ada_score_cross,br_score,br_score_cross,gb_score,gb_score_cross]},index=name)
 go['RMSE']=go.applymap(lambda x: x.mean())
 go.RMSE.sort_values()
```
![image](https://user-images.githubusercontent.com/65950195/151909520-6b7c5135-1df4-43fe-ba8a-4b561b3f1ede.png)

## Conclusion
As Gradient boosting has less RMSE value, we use this model to predict our test data output
```bash
 predict=gb.predict(test_dummy)
 df=pd.DataFrame({'Item_Outlet_Sales':predict})
 corr_ans=pd.concat([sample,df],axis=1)
 del corr_ans['Unnamed: 0']
 corr_ans
```
