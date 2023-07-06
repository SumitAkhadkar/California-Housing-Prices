### INTRODUCTION:
  This dataset contains the brief details of California Housing data in which various details is mention related to housing price. By analysing this dataset, we will be able to predict the approximate price for the various houses. The main aim is to find out the mean price of houses based on locations. This dataset contains various features which is as follows:



#### Features:
1.Longitude: It is a horizontal distance of a house measure from west to east. Higher value indicates it is further towards west. It is a quantitative data of continuous type.

2.Latitude: It is a vertical distance of house measure form north to south. Higher value indicates it is further towards north. It is a quantitative data of continuous type.

3.housing_median_age: It is median age of a house. It is quantitative data of continuous type.

4.total_rooms: Total number of rooms in a block. It is quantitative data of discrete type.

5.total_bedrooms: Total number of bedrooms in a block. It is quantitative data of discrete type.

6.Population: Total number of people residing in a block. It is quantitative data of discrete type.

7.households: Total number of households, a group of people residing within a home. It is quantitative data of discrete type.

8.median_income: It is median income of household in US Dollars. It is a quantitative data of continuous type.

9.median_house_value: It is median value of household in US Dollars. It is a quantitative data of continuous type.

10.ocean_proximity: It gives location of the house with respect to ocean. It is of categorical data of nominal data type.



```python
# importing required library for data analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

```


```python
df = pd.read_excel('housing.xlsx')         #file reading
df                                         # print output
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41</td>
      <td>880</td>
      <td>129.0</td>
      <td>322</td>
      <td>126</td>
      <td>8.3252</td>
      <td>452600</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21</td>
      <td>7099</td>
      <td>1106.0</td>
      <td>2401</td>
      <td>1138</td>
      <td>8.3014</td>
      <td>358500</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52</td>
      <td>1467</td>
      <td>190.0</td>
      <td>496</td>
      <td>177</td>
      <td>7.2574</td>
      <td>352100</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1274</td>
      <td>235.0</td>
      <td>558</td>
      <td>219</td>
      <td>5.6431</td>
      <td>341300</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1627</td>
      <td>280.0</td>
      <td>565</td>
      <td>259</td>
      <td>3.8462</td>
      <td>342200</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20635</th>
      <td>-121.09</td>
      <td>39.48</td>
      <td>25</td>
      <td>1665</td>
      <td>374.0</td>
      <td>845</td>
      <td>330</td>
      <td>1.5603</td>
      <td>78100</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>-121.21</td>
      <td>39.49</td>
      <td>18</td>
      <td>697</td>
      <td>150.0</td>
      <td>356</td>
      <td>114</td>
      <td>2.5568</td>
      <td>77100</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>-121.22</td>
      <td>39.43</td>
      <td>17</td>
      <td>2254</td>
      <td>485.0</td>
      <td>1007</td>
      <td>433</td>
      <td>1.7000</td>
      <td>92300</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>-121.32</td>
      <td>39.43</td>
      <td>18</td>
      <td>1860</td>
      <td>409.0</td>
      <td>741</td>
      <td>349</td>
      <td>1.8672</td>
      <td>84700</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>-121.24</td>
      <td>39.37</td>
      <td>16</td>
      <td>2785</td>
      <td>616.0</td>
      <td>1387</td>
      <td>530</td>
      <td>2.3886</td>
      <td>89400</td>
      <td>INLAND</td>
    </tr>
  </tbody>
</table>
<p>20640 rows × 10 columns</p>
</div>




```python
df.shape          # This will indicates the total no. of rows and columns in the dataset
```




    (20640, 10)



1.What is the average median income of the data set and check the distribution of data using appropriate plots. 
Please explain the distribution of the plot.


```python
df1 = df['median_income'].mean()       # mean() function gives the mean value of particular column.
df1
```




    3.8706710029069766



From above output, It indicates that average median income of the dataset is 3.8706710 which is approximately is 3.9


```python
df.hist(bins=40,figsize=(20,20))                    # Histogram is used to see the distribution of a numerical value
plt.show()
```


    
![png](output_8_0.png)
    


From the above plot, it is to be noted that total_rooms,total_bedrooms,population,households,median_income are of Right skewed.

while logitude and latitude are of asymmetric,i.e.,highly skewed 

and While housing_median_age and for median_house_value have some outliers.

2.Draw an appropriate plot to see the distribution of housing_median_age and explain your observations.


```python
plt.figure(figsize=(5,3))                                    # To fix the figure size
plt.hist(df['housing_median_age'],bins = 30)                 # Histogram is used to see the distribution of a numerical value.
plt.grid()                                                   # It is used to show grid lines
plt.title('Histogram plot of Housing_median_age')            # To title the plot
plt.xlabel('housing median age')                             # To label x-axis as housing median age 
plt.ylabel('Frequencies')                                    # To label y-axis as Frequencies
plt.show()
```


    
![png](output_11_0.png)
    


From above plot, we can come to analysis that it is distributed symmetrically.

The skewness of this plot can be found out by using formula : Skewed =3*(mean-median)/std()


```python
mean_df = df['housing_median_age'].mean()          # to find mean
mean_df
```




    28.639486434108527




```python
median_df = df['housing_median_age'].median()      # to find median
median_df
```




    29.0




```python
std_df = df['housing_median_age'].std()            # to find std. deviation
std_df
```




    12.585557612111637




```python
Skewed = (3*(mean_df-median_df))/std_df            # To find skewness of plot
Skewed
```




    -0.08593506390480511



The Value of skewness is -0.08 which is between -0.5 and 0.5. hence, that data is perfectly symmetrical.

So, from the histogram and from the Skewness it is to be noted that the housing_median_age is perfectly symmetrical.

3.Show with the help of visualization, how median_income and median_house_values are related?


```python
plt.figure(figsize=(6,4))
sns.scatterplot(x='median_house_value', y='median_income', data = df)  #Scatter plot gives relation between two numerical values
plt.show()                                                             # X-axis = median_house_value & Y-axis = median_income
```


    
![png](output_19_0.png)
    


From the above visualisation it is to be analysed that with an increase in the median_house_value there is also an increase in the median income. Hence, median_house_value is directly proportional to median income.

While, some outliers is present in median_house_value which is shown in the figure.


4.Create a data set by deleting the corresponding examples from the data set for which total_bedrooms are not available.


```python
df[df.isnull().any(axis=1)]     # isnull() return the boolean value True for NULL values, and otherwise False.
                                # Here missing values are denoted by NaN
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>290</th>
      <td>-122.16</td>
      <td>37.77</td>
      <td>47</td>
      <td>1256</td>
      <td>NaN</td>
      <td>570</td>
      <td>218</td>
      <td>4.3750</td>
      <td>161900</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>341</th>
      <td>-122.17</td>
      <td>37.75</td>
      <td>38</td>
      <td>992</td>
      <td>NaN</td>
      <td>732</td>
      <td>259</td>
      <td>1.6196</td>
      <td>85100</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>538</th>
      <td>-122.28</td>
      <td>37.78</td>
      <td>29</td>
      <td>5154</td>
      <td>NaN</td>
      <td>3741</td>
      <td>1273</td>
      <td>2.5762</td>
      <td>173400</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>563</th>
      <td>-122.24</td>
      <td>37.75</td>
      <td>45</td>
      <td>891</td>
      <td>NaN</td>
      <td>384</td>
      <td>146</td>
      <td>4.9489</td>
      <td>247100</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>696</th>
      <td>-122.10</td>
      <td>37.69</td>
      <td>41</td>
      <td>746</td>
      <td>NaN</td>
      <td>387</td>
      <td>161</td>
      <td>3.9063</td>
      <td>178400</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20267</th>
      <td>-119.19</td>
      <td>34.20</td>
      <td>18</td>
      <td>3620</td>
      <td>NaN</td>
      <td>3171</td>
      <td>779</td>
      <td>3.3409</td>
      <td>220500</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20268</th>
      <td>-119.18</td>
      <td>34.19</td>
      <td>19</td>
      <td>2393</td>
      <td>NaN</td>
      <td>1938</td>
      <td>762</td>
      <td>1.6953</td>
      <td>167400</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20372</th>
      <td>-118.88</td>
      <td>34.17</td>
      <td>15</td>
      <td>4260</td>
      <td>NaN</td>
      <td>1701</td>
      <td>669</td>
      <td>5.1033</td>
      <td>410700</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>20460</th>
      <td>-118.75</td>
      <td>34.29</td>
      <td>17</td>
      <td>5512</td>
      <td>NaN</td>
      <td>2734</td>
      <td>814</td>
      <td>6.6073</td>
      <td>258100</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>20484</th>
      <td>-118.72</td>
      <td>34.28</td>
      <td>17</td>
      <td>3051</td>
      <td>NaN</td>
      <td>1705</td>
      <td>495</td>
      <td>5.7376</td>
      <td>218600</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
<p>207 rows × 10 columns</p>
</div>



In the above code,missing values are identified by using isnull()method for column total_bedrooms


```python
df1= df.dropna()                          # dropna() is used to delete the records which have none value
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41</td>
      <td>880</td>
      <td>129.0</td>
      <td>322</td>
      <td>126</td>
      <td>8.3252</td>
      <td>452600</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21</td>
      <td>7099</td>
      <td>1106.0</td>
      <td>2401</td>
      <td>1138</td>
      <td>8.3014</td>
      <td>358500</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52</td>
      <td>1467</td>
      <td>190.0</td>
      <td>496</td>
      <td>177</td>
      <td>7.2574</td>
      <td>352100</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1274</td>
      <td>235.0</td>
      <td>558</td>
      <td>219</td>
      <td>5.6431</td>
      <td>341300</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1627</td>
      <td>280.0</td>
      <td>565</td>
      <td>259</td>
      <td>3.8462</td>
      <td>342200</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20635</th>
      <td>-121.09</td>
      <td>39.48</td>
      <td>25</td>
      <td>1665</td>
      <td>374.0</td>
      <td>845</td>
      <td>330</td>
      <td>1.5603</td>
      <td>78100</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>-121.21</td>
      <td>39.49</td>
      <td>18</td>
      <td>697</td>
      <td>150.0</td>
      <td>356</td>
      <td>114</td>
      <td>2.5568</td>
      <td>77100</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>-121.22</td>
      <td>39.43</td>
      <td>17</td>
      <td>2254</td>
      <td>485.0</td>
      <td>1007</td>
      <td>433</td>
      <td>1.7000</td>
      <td>92300</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>-121.32</td>
      <td>39.43</td>
      <td>18</td>
      <td>1860</td>
      <td>409.0</td>
      <td>741</td>
      <td>349</td>
      <td>1.8672</td>
      <td>84700</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>-121.24</td>
      <td>39.37</td>
      <td>16</td>
      <td>2785</td>
      <td>616.0</td>
      <td>1387</td>
      <td>530</td>
      <td>2.3886</td>
      <td>89400</td>
      <td>INLAND</td>
    </tr>
  </tbody>
</table>
<p>20433 rows × 10 columns</p>
</div>



While in the above code the missing values are dropped from the column named'total_bedrooms'by using dropna() method.

5. Create a data set by filling the missing data with the mean value of the total_bedrooms in the original data set.


```python
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41</td>
      <td>880</td>
      <td>129.0</td>
      <td>322</td>
      <td>126</td>
      <td>8.3252</td>
      <td>452600</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21</td>
      <td>7099</td>
      <td>1106.0</td>
      <td>2401</td>
      <td>1138</td>
      <td>8.3014</td>
      <td>358500</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52</td>
      <td>1467</td>
      <td>190.0</td>
      <td>496</td>
      <td>177</td>
      <td>7.2574</td>
      <td>352100</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1274</td>
      <td>235.0</td>
      <td>558</td>
      <td>219</td>
      <td>5.6431</td>
      <td>341300</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1627</td>
      <td>280.0</td>
      <td>565</td>
      <td>259</td>
      <td>3.8462</td>
      <td>342200</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20635</th>
      <td>-121.09</td>
      <td>39.48</td>
      <td>25</td>
      <td>1665</td>
      <td>374.0</td>
      <td>845</td>
      <td>330</td>
      <td>1.5603</td>
      <td>78100</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>-121.21</td>
      <td>39.49</td>
      <td>18</td>
      <td>697</td>
      <td>150.0</td>
      <td>356</td>
      <td>114</td>
      <td>2.5568</td>
      <td>77100</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>-121.22</td>
      <td>39.43</td>
      <td>17</td>
      <td>2254</td>
      <td>485.0</td>
      <td>1007</td>
      <td>433</td>
      <td>1.7000</td>
      <td>92300</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>-121.32</td>
      <td>39.43</td>
      <td>18</td>
      <td>1860</td>
      <td>409.0</td>
      <td>741</td>
      <td>349</td>
      <td>1.8672</td>
      <td>84700</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>-121.24</td>
      <td>39.37</td>
      <td>16</td>
      <td>2785</td>
      <td>616.0</td>
      <td>1387</td>
      <td>530</td>
      <td>2.3886</td>
      <td>89400</td>
      <td>INLAND</td>
    </tr>
  </tbody>
</table>
<p>20640 rows × 10 columns</p>
</div>



In the above code, a new dataset had been created where the missing values in the 'total_bedrooms' which are denoted by NaN are replaced with the mean value of the 'total_bedrooms'. i.e., 537.870553

6.Write a programming construct (create a user defined function) to calculate the median value of the data set wherever required.

##### MEDIAN():
* It is defined as the 50th percentile of the set of measurements, when observed measurements are ranked from smallest to highest. 
* It is not Sensitive to Outliers. 
* It is Commonly used by Discrete and Continuous data.
* It can be used as a summary measure for ordinal observation.



```python
def median1(n):                          # define the user function for median
    median = sorted(n)[len(n)//2]
    return median                        # Return the value of median expression
```


```python
print(median1(df['longitude']))
print(median1(df['latitude']))
print(median1(df['housing_median_age']))
print(median1(df['total_rooms']))
print(median1(df['total_bedrooms'])) 
print(median1(df['population']))
print(median1(df['households']))
print(median1(df['median_income']))
print(median1(df['median_house_value']))
```

    -118.49
    34.26
    29
    2127
    438.0
    1166
    409
    3.5349
    179700
    

By calling user defined function we calculate the median value of the each numerical data.

7. Plot latitude versus longitude and explain your observations.


```python
plt.figure(figsize=(6,4))
sns.scatterplot(x='latitude', y='longitude', data = df)              #Scatter plot gives relation between two numerical values
plt.show()                                                           # X-axis = latitude & Y-axis = longitude
```


    
![png](output_35_0.png)
    


* From the above plot, it is to be noted that with an increase in latitude, longitude is decreased. 
* From this we can say that longitude is inversely proportional to latitude.
* From the above plot it is to be noted that latitude vs longitude has negative correlation as here y-axis is increasing while x-axis is decreasing., both are moving in an opposite direction.


8.Create a data set for which the ocean_proximity is ‘Near ocean’.


```python
new_df = df[df.ocean_proximity == 'NEAR OCEAN']   # This will filter the data set having ocean_proximity = 'NEAR OCEAN'
new_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1850</th>
      <td>-124.17</td>
      <td>41.80</td>
      <td>16</td>
      <td>2739</td>
      <td>480.0</td>
      <td>1259</td>
      <td>436</td>
      <td>3.7557</td>
      <td>109400</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>1851</th>
      <td>-124.30</td>
      <td>41.80</td>
      <td>19</td>
      <td>2672</td>
      <td>552.0</td>
      <td>1298</td>
      <td>478</td>
      <td>1.9797</td>
      <td>85800</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>1852</th>
      <td>-124.23</td>
      <td>41.75</td>
      <td>11</td>
      <td>3159</td>
      <td>616.0</td>
      <td>1343</td>
      <td>479</td>
      <td>2.4805</td>
      <td>73200</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>1853</th>
      <td>-124.21</td>
      <td>41.77</td>
      <td>17</td>
      <td>3461</td>
      <td>722.0</td>
      <td>1947</td>
      <td>647</td>
      <td>2.5795</td>
      <td>68400</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>1854</th>
      <td>-124.19</td>
      <td>41.78</td>
      <td>15</td>
      <td>3140</td>
      <td>714.0</td>
      <td>1645</td>
      <td>640</td>
      <td>1.6654</td>
      <td>74600</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20380</th>
      <td>-118.83</td>
      <td>34.14</td>
      <td>16</td>
      <td>1316</td>
      <td>194.0</td>
      <td>450</td>
      <td>173</td>
      <td>10.1597</td>
      <td>500001</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20381</th>
      <td>-118.83</td>
      <td>34.14</td>
      <td>16</td>
      <td>1956</td>
      <td>312.0</td>
      <td>671</td>
      <td>319</td>
      <td>6.4001</td>
      <td>321800</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20423</th>
      <td>-119.00</td>
      <td>34.08</td>
      <td>17</td>
      <td>1822</td>
      <td>438.0</td>
      <td>578</td>
      <td>291</td>
      <td>5.4346</td>
      <td>428600</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20424</th>
      <td>-118.75</td>
      <td>34.18</td>
      <td>4</td>
      <td>16704</td>
      <td>2704.0</td>
      <td>6187</td>
      <td>2207</td>
      <td>6.6122</td>
      <td>357600</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20425</th>
      <td>-118.75</td>
      <td>34.17</td>
      <td>18</td>
      <td>6217</td>
      <td>858.0</td>
      <td>2703</td>
      <td>834</td>
      <td>6.8075</td>
      <td>325900</td>
      <td>NEAR OCEAN</td>
    </tr>
  </tbody>
</table>
<p>2658 rows × 10 columns</p>
</div>



9. Find the mean and median of the median income for the data set created in question 8


```python
new_df['median_income'].mean()          # mean() gives the averages of the data
```




    4.0057848006019565




```python
new_df['median_income'].median()       # median() value gives the 50th percentile of the set of all observations.
```




    3.64705



The mean and median value of the 'median_income' as per new dataset are:4.005784 and 3.64705 respectively.

10.Please create a new column named total_bedroom_size. If the total bedrooms is 10 or less, it should be quoted as small. If the total bedrooms is 11 or more but less than 1000, it should be medium, otherwise it should be considered large.


```python
list1 = [(df['total_bedrooms'] <=10),
         (df['total_bedrooms'] >=11)&(df['total_bedrooms'] <=1000),
         (df['total_bedrooms'] > 1000)]

values = ['small','medium','large']

df['total_bedroom_size']= np.select(list1,values)

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
      <th>total_bedroom_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41</td>
      <td>880</td>
      <td>129.0</td>
      <td>322</td>
      <td>126</td>
      <td>8.3252</td>
      <td>452600</td>
      <td>NEAR BAY</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21</td>
      <td>7099</td>
      <td>1106.0</td>
      <td>2401</td>
      <td>1138</td>
      <td>8.3014</td>
      <td>358500</td>
      <td>NEAR BAY</td>
      <td>large</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52</td>
      <td>1467</td>
      <td>190.0</td>
      <td>496</td>
      <td>177</td>
      <td>7.2574</td>
      <td>352100</td>
      <td>NEAR BAY</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1274</td>
      <td>235.0</td>
      <td>558</td>
      <td>219</td>
      <td>5.6431</td>
      <td>341300</td>
      <td>NEAR BAY</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1627</td>
      <td>280.0</td>
      <td>565</td>
      <td>259</td>
      <td>3.8462</td>
      <td>342200</td>
      <td>NEAR BAY</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20635</th>
      <td>-121.09</td>
      <td>39.48</td>
      <td>25</td>
      <td>1665</td>
      <td>374.0</td>
      <td>845</td>
      <td>330</td>
      <td>1.5603</td>
      <td>78100</td>
      <td>INLAND</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>-121.21</td>
      <td>39.49</td>
      <td>18</td>
      <td>697</td>
      <td>150.0</td>
      <td>356</td>
      <td>114</td>
      <td>2.5568</td>
      <td>77100</td>
      <td>INLAND</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>-121.22</td>
      <td>39.43</td>
      <td>17</td>
      <td>2254</td>
      <td>485.0</td>
      <td>1007</td>
      <td>433</td>
      <td>1.7000</td>
      <td>92300</td>
      <td>INLAND</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>-121.32</td>
      <td>39.43</td>
      <td>18</td>
      <td>1860</td>
      <td>409.0</td>
      <td>741</td>
      <td>349</td>
      <td>1.8672</td>
      <td>84700</td>
      <td>INLAND</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>-121.24</td>
      <td>39.37</td>
      <td>16</td>
      <td>2785</td>
      <td>616.0</td>
      <td>1387</td>
      <td>530</td>
      <td>2.3886</td>
      <td>89400</td>
      <td>INLAND</td>
      <td>medium</td>
    </tr>
  </tbody>
</table>
<p>20640 rows × 11 columns</p>
</div>



* In the above data set a new column named total_bedroom_size had been added. Where the total_bedroom_size had been compared to the total_bedrooms while mentioning about the sizes of the total_bedroom_size, 
where:
* If the total_bedroom_size <=10 it is indicated as "small" 
* If the total_bedroom_size >=11 or <=1000 it is indicated as "medium"
* And  If the total_bedroom_size >1000 it is indicated as "large"


#### CONCLUSION: 

* From the given dataset it is to be noted that the outliers are present for housing_median_age and median_house_value.
* All the features present in this dataset are of numerical type except ocean_proximity which is of categorical data type.
* housing_median_age is of fairly symmetric as the skewness value is -0.08 which is between -0.5 and 0.5.
* Plot between median_income vs median_house_values is directly proportional to each other as it has positive correlation.
* Plot between latitude and longitude is inversely proportional to each other and it has a negative correlation.

