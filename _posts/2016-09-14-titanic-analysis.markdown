
### Description[Titanic](https://www.kaggle.com/c/titanic)
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

**<font color=red>Before analyzing the data, I want to see the data first.</font>**<br>
Using pandas to load the data, and show some information of the data.


```python
import pandas as pd 
import numpy as np 
from pandas import Series, DataFrame

data_titanic = pd.read_csv("./titanic_data.csv")
data_titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    

Obviously, there are some information I can get.
+ 891 passangers in total
+ Age some data lost, only 714
+ Cabin even less, only 204
<br>But I want to see more about the data, so I use the built-in function describe to show me the description:


```python
data_titanic.describe()
```

    C:\Anaconda2\lib\site-packages\numpy\lib\function_base.py:3834: RuntimeWarning: Invalid value encountered in percentile
      RuntimeWarning)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



There is a RuntimeWarning when I run the built-in function describe. Because the loss of the data, I can't get the 25%, 50% and 75% number of Age. So I should do something to fix it, but anyway, there still are some useful information.
<br>From the mean feild, I find that the probability survived at last is 0.383838, and the number of first-class is less than other two class.
<br>Now I want to analyze the connection between single attribution(or conbination) and the probability of survival(It is the most important thing!).
<br>SO I POST MY QUESTION:
<br><font color=red font=20 bold>WHAT MATTERS WHEN YOU WANT TO SURVIVE FROM TITANIC?</font>


```python
%pylab inline
import matplotlib.pyplot as plt
fig = plt.figure()

fig, axes = plt.subplots(nrows=1, ncols=3)

#Draw graphs in a single one
Survived_0 = data_titanic.Pclass[data_titanic.Survived == 0].value_counts()
Survived_1 = data_titanic.Pclass[data_titanic.Survived == 1].value_counts()
df1=pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df1.plot(kind='bar', ax=axes[0], figsize=(12, 4), title="Survival with different Pclass(Population)", stacked=True)


Survived_0 = data_titanic.Sex[data_titanic.Survived == 0].value_counts()
Survived_1 = data_titanic.Sex[data_titanic.Survived == 1].value_counts()
df2=pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df2.plot(kind='bar', ax=axes[1], figsize=(12, 4), title="Survival with different Sex(Population)", stacked=True)


Survived_0 = data_titanic.Embarked[data_titanic.Survived == 0].value_counts()
Survived_1 = data_titanic.Embarked[data_titanic.Survived == 1].value_counts()
df3=pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df3.plot(kind='bar', ax=axes[2], figsize=(12, 4), title="Survival with different Embarked(Population)", stacked=True)

plt.tight_layout()
```

    Populating the interactive namespace from numpy and matplotlib
    

    WARNING: pylab import has clobbered these variables: ['axes']
    `%matplotlib` prevents importing * from pylab and numpy
    


    <matplotlib.figure.Figure at 0x1553fda0>



![png](output_6_3.png)


I make three figures about survival from Pclass, Sex and Embarked(because the ranges of Age, Sibsp, Parch and Fare are so large that I should deal with the data first)
+ I find that the number of third-class is more than other two classes, and the probability of death in third-class is much higher than it in other two classes too; 
+ the population of male is almost double of female, while the survived population is on the contrary, it's more prosible for a woman surviving at last, (Lady first! So gentle British people are!).
+ There were more people embarking from S port, but I can't find the relationship between the probability of survival and the embarkation port.

<font color=blue>First, I want to deal with the data of Age.
<br>I want to see the the distribution of passangers' age, so I draw the dense graph as follows:


```python
data_titanic.Age.plot(kind='kde')
plt.xlabel(u"Age")
plt.ylabel(u"Density") 
plt.title(u"The distribution of passangers' age")
plt.show()
```


![png](output_9_0.png)


As we can see, most people are among 20 to 40, and the probability near 30 is the highest. From the age group devided in [this link](http://www.ncbi.nlm.nih.gov/pubmed/11815703), I set age into 4 groups like youth(0-17), young adults(18-35), middle-aged adults(36-55) and older adults(56+)


```python
def age_adjusted(age):
    if age < 18:
        return 'Youth'
    elif age < 36:
        return 'Young Adults'
    elif age < 56:
        return 'Mid-age Adults'
    elif age >= 56:
        return 'Older Adults'

age_group = pd.DataFrame(data_titanic.Age.apply(age_adjusted))
age_group.columns = ['GroupAge']

titanic_final = pd.concat([data_titanic, age_group], axis=1)

Survived_0 = titanic_final.GroupAge[titanic_final.Survived == 0].value_counts()
Survived_1 = titanic_final.GroupAge[titanic_final.Survived == 1].value_counts()
df_age=pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df_age.plot(kind='bar', title="Survival with different groups of Age(Population)", stacked=True)
plt.show()
```


![png](output_11_0.png)


It seems that youth people get higher opportunities to survive at last, so age is a notable character, it should be discussed deeper later.

<font color=blue>Second, I want to deal with the data of Sibsp.</font>
<br>Maybe there is a connection between the number of sibsp and survival, so I grouped it.


```python
g_survived = data_titanic[data_titanic.Survived == 1].groupby(['SibSp'])
g_total = data_titanic.groupby(['SibSp'])
df = pd.DataFrame(g_survived.count()['PassengerId'] / g_total.count()['PassengerId']).fillna(value=0)
df.PassengerId.plot(kind='bar')
plt.xlabel(u"Sibsp")
plt.ylabel(u"Probability of Survival") 
plt.title(u"Passangers' survival rate with different Sibsp")
plt.show()
```


![png](output_14_0.png)


From the diagram above, we can see that with the precondition of having sibsp, the number of SibSp growing, the probability of survival is getting less, maybe just because those having many sibsp considering too much in the disaster, they wanted to find their family first that they spent too much time for survival. 
<br><font color=blue>Third, I deal with the data of Parch.</font>


```python
g_survived = data_titanic[data_titanic.Survived == 1].groupby(['Parch'])
g_total = data_titanic.groupby(['Parch'])
df = pd.DataFrame(g_survived.count()['PassengerId'] / g_total.count()['PassengerId']).fillna(value=0)
df.PassengerId.plot(kind='bar')
plt.xlabel(u"Parch")
plt.ylabel(u"Probability of Survival") 
plt.title(u"Passangers' survival rate with different Parch")
plt.show()
```


![png](output_16_0.png)


It's quite difficult to see the relationship, so I skipped this one.
<br><font color=blue>Then I deal with Fare. With a high standard deviation, so I want to standardize data first.</font> [use preprocessing.StandardScaler()](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)


```python
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
fare_scale_param = scaler.fit(titanic_final['Fare'])
titanic_final['FareScaled'] = scaler.fit_transform(titanic_final['Fare'], fare_scale_param)
plt.scatter(titanic_final.Survived, titanic_final.FareScaled)
plt.ylabel(u"Standardized Fare")
plt.title(u"Survival with Standardized Fare")
plt.grid(b=True, which='major', axis='y')
```

    C:\Anaconda2\lib\site-packages\sklearn\preprocessing\data.py:583: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Anaconda2\lib\site-packages\sklearn\preprocessing\data.py:583: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Anaconda2\lib\site-packages\sklearn\preprocessing\data.py:646: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    


![png](output_18_1.png)


It seems there is no significant connection between Fare and Survival.
<br><font color=blue>Finally I deal with Cabin. Since the absence of information, no significant connection between survival and Cabin number in addition, I set Cabin 'Yes' or 'No' to identify the relationship.


```python
def set_cabin(data):
    data.loc[(data.Cabin.notnull()), 'Cabin'] = "Yes"
    data.loc[(data.Cabin.isnull()), 'Cabin'] = "No"
    return data

titanic_final = set_cabin(titanic_final) # only be run once, if run again, the function will set 'No' to 'Yes'
```


```python
Survived_0 = titanic_final.Cabin[titanic_final.Survived == 0].value_counts()
Survived_1 = titanic_final.Cabin[titanic_final.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df.plot(kind='bar', stacked=True, title='Survival with having Cabin or Not(Population)')

plt.show()
```


![png](output_21_0.png)


From the chart, I found that with a Cabin, the survival probability is higher.
From the above analysis with a single attribute, I found that if a person met the condition as follows, he/she would get higher probability to survival
+ a first-class ticket
+ a female
+ was young 
+ had a Cabin
+ had few Sibsp(While the data is small when one had more sibsp, so there was random error)
<br><font color=red>Now, I want to conbine some attributes to make a more significate result.</font>
<br>How is the conbination of Pclass and Sex going?
<br>How is Pclass and Age?
<br>And Pclass with a Cabin?


```python
fig, axes = plt.subplots(nrows=3, ncols=4)

#Draw graphs in a single one
df0 = pd.DataFrame(titanic_final.Survived[titanic_final.Sex == 'female'][titanic_final.Pclass == 1].value_counts())
df0.columns = ['High-class Female']
df0.plot(kind='bar', ax=axes[0][0], figsize=(24, 16), color='r', title="High-class Female(Population)")

df1 = pd.DataFrame(titanic_final.Survived[titanic_final.Sex == 'female'][titanic_final.Pclass != 1].value_counts())
df1.columns = ['Low-class Female']
df1.plot(kind='bar', ax=axes[0][1], figsize=(24, 16), color='r', title="Low-class Female(Population)")

df2 = pd.DataFrame(titanic_final.Survived[titanic_final.Sex == 'male'][titanic_final.Pclass == 1].value_counts())
df2.columns = ['High-class Male']
df2.plot(kind='bar', ax=axes[0][2], figsize=(24, 16), color='r', title="High-class Male(Population)")

df3 = pd.DataFrame(titanic_final.Survived[titanic_final.Sex == 'male'][titanic_final.Pclass != 1].value_counts())
df3.columns = ['Low-class Male']
df3.plot(kind='bar', ax=axes[0][3], figsize=(24, 16), color='r', title="Low-class Male(Population)")

df4 = pd.DataFrame(titanic_final.Survived[titanic_final.GroupAge == 'Youth'][titanic_final.Pclass == 1].value_counts())
df4.columns = ['High-class Youth']
df4.plot(kind='bar', ax=axes[1][0], figsize=(24, 16), color='g', title="High-class Youth(Population)")

df5 = pd.DataFrame(titanic_final.Survived[titanic_final.GroupAge == 'Youth'][titanic_final.Pclass != 1].value_counts())
df5.columns = ['Low-class Youth']
df5.plot(kind='bar', ax=axes[1][1], figsize=(24, 16), color='g', title="Low-class Youth(Population)")

df6 = pd.DataFrame(titanic_final.Survived[titanic_final.GroupAge != 'Youth'][titanic_final.Pclass == 1].value_counts())
df6.columns = ['High-class Non-Youth']
df6.plot(kind='bar', ax=axes[1][2], figsize=(24, 16), color='g', title="High-class Non-Youth(Population)")

df7 = pd.DataFrame(titanic_final.Survived[titanic_final.GroupAge != 'Youth'][titanic_final.Pclass != 1].value_counts())
df7.columns = ['Low-class Non-Youth']
df7.plot(kind='bar', ax=axes[1][3], figsize=(24, 16), color='g', title="Low-class Non-Youth(Population)")

df8 = pd.DataFrame(titanic_final.Survived[titanic_final.Cabin == 'Yes'][titanic_final.Pclass == 1].value_counts())
df8.columns = ['High-class Cabin']
df8.plot(kind='bar', ax=axes[2][0], figsize=(24, 16), color='b', title="High-class Cabin(Population)")

df9 = pd.DataFrame(titanic_final.Survived[titanic_final.Cabin == 'Yes'][titanic_final.Pclass != 1].value_counts())
df9.columns = ['Low-class Cabin']
df9.plot(kind='bar', ax=axes[2][1], figsize=(24, 16), color='b', title="Low-class Cabin(Population)")

df10 = pd.DataFrame(titanic_final.Survived[titanic_final.Cabin == 'No'][titanic_final.Pclass == 1].value_counts())
df10.columns = ['High-class Non-Cabin']
df10.plot(kind='bar', ax=axes[2][2], figsize=(24, 16), color='b', title="High-class Non-Cabin(Population)")

df11 = pd.DataFrame(titanic_final.Survived[titanic_final.Cabin == 'No'][titanic_final.Pclass != 1].value_counts())
df11.columns = ['Low-class Non-Cabin']
df11.plot(kind='bar', ax=axes[2][3], figsize=(24, 16), color='b', title="Low-class Non-Cabin(Population)")

plt.tight_layout()
```


![png](output_23_0.png)


From the above graphs, we see that the advantages of attribution is superposed, while Female with first-class ticket or young people with first-class ticket has the highest probability to survive.
