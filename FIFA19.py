import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.simplefilter("ignore")


# Importing data
data = pd.read_csv("C:/Users/jubin/Dropbox/A.I/data.csv", index_col =0)
data_sel1 = data.iloc[1:10,[83,84,85,86,87]]
# Finding the null values:
null_val = data.isna().sum()
print(data.shape)

# Data visualization for understanding the data

# Comparison of preferred foot over the different p layers
data['Preferred Foot'].value_counts().head(50).plot.bar(color= 'purple')

# Comparison of international rep
data['International Reputation'].value_counts()

# plotting a pie chart to represent share of international reputation

sizes = [16532, 1261, 309, 51, 6]
colors = ['red', 'yellow', 'green', 'pink', 'blue']

explode = [0.1, 0.1, 0.2, 0.5, 0.9]

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)
plt.title('A Pie Chart for International Repuatation for the Football Players', fontsize = 25)
plt.legend()
plt.show()

# plotting a pie chart to represent the share of week foot players
# With this we can find out the which player has better higher shot power and ball control

data['Weak Foot'].value_counts()

# Plotting a pie chart to represent the share of week foot player
labels = ['5', '4', '3', '2', '1']

size = [229, 2262, 11349, 3761, 158]
colors = ['red', 'yellow', 'green', 'pink', 'blue']
explode = [0.1, 0.1, 0.1, 0.1, 0.1]

plt.pie(size, labels= labels, colors = colors, explode = explode, shadow = True)
plt.title('Pie chart for Representing Week Foot of the players', fontsize= 25)
plt.legend()
plt.show()

# different positions acquired by the players

plt.figure(figsize=(12,8))
sns.set(style = 'dark', palette= 'colorblind', color_codes= True)
ax = sns.countplot('Position', data = data, color = 'orange')
ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of players', fontsize = 16)
ax.set_title(label = 'Comparison of positions and players', fontsize = 20)

plt.show()

# defining a function for cleaning the Weight data

def ext_val(value):
    out = value.replace('lbs','')
    return float(out)

# applying the function to wight column

data['Weight'].fillna('200lbs', inplace = True)
data['Weight'] = data['Weight'].apply(lambda x: ext_val(x))

data['Weight'].head()

# defining a function to transform the wage and value column
def trs_wa_val(value):
    out = value.replace('€','')
    if 'M' in out:
        out = float(out.replace('M',''))*1000000
    elif 'K' in out:
        out = float(out.replace('K',''))*1000
    return float(out)

# applying the function to value and wage column

data['Value'] = data['Value'].apply(lambda x: trs_wa_val(x))
data['Wage'] = data['Wage'].apply(lambda x: trs_wa_val(x))

data['Wage'].head()

# Skill Moves of Players
plt.figure(figsize =(7, 8))
ax = sns.countplot(x ='Skill Moves', data = data, palette ='pastel')
ax.set_title(label = 'Count of players on Basis of their skill moves', fontsize = 20)
ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()

# Showing different potential scores of the players participating in the FIFA 2019

sns.set(style= "dark", palette= "muted", color_codes= True)
x = data.Potential
plt.figure(figsize=(12, 8))
ax = sns.distplot(x, bins= 58, kde= False, color= 'y')
ax.set_xlabel(xlabel= "Player's Potential Scores", fontsize = 16)
ax.set_ylabel(ylabel= 'Number of players', fontsize = 16)
ax.set_title(label ='Histogram of players Potential Scores', fontsize = 20)
plt.show()

# Showing different overall scores of the players participating in the FIFA 2019

sns.set(style= "dark", palette= "deep", color_codes= True)
x = data.Overall
plt.figure(figsize= (12, 8))
ax =sns.distplot(x, bins= 52, kde = False, color= 'r')
ax.set_xlabel(xlabel= "Player\'s Scores", fontsize= 16)
ax.set_ylabel(ylabel= "Number of players", fontsize = 16)
ax.set_title(label = 'Histogram of players Overall Scores', fontsize = 20)
plt.show()



# Dropping unrequired columns

data.drop(['ID', 'Name','Age', 'Photo','Nationality', 'Flag', 'Overall',
           'Potential','Club','Club Logo', 'Value', 'Wage','Special', 'Preferred Foot',
           'International Reputation', 'Weak Foot', 'Skill Moves', 'Work Rate', 'Body Type',
           'Real Face', 'Position', 'Jersey Number', 'Joined',
           'Loaned From', 'Contract Valid Until', 'Height', 'Weight', 'Release Clause'], axis=1, inplace=True)

# Dropping rows with missing data
data.dropna(axis=0, how='any', inplace=True)

# Correlation matrix & Heatmap
pl.figure(figsize =(20,20))
corrmat = data.corr()
sns.heatmap(corrmat, annot=True, fmt='.1f', vmin=0, vmax=1, square=True);

# Dropping uncorrelated columns
data.drop(columns=['GKDiving','GKHandling','GKKicking',
                   'GKPositioning','GKReflexes','LS','ST',
                   'RS','LW','LF','CF','RF','RW','LAM','CAM',
                   'RAM','LM','LCM','CM','RCM','RM','LWB','LDM',
                   'CDM','RDM','RWB','LB','LCB','CB','RCB','RB'], inplace=True)
data.columns
data.shape

#Calculating the mean value of the whole featureset
cols = ['Crossing','Finishing','HeadingAccuracy',
        'ShortPassing','Volleys','Dribbling','Curve',
        'FKAccuracy','LongPassing','BallControl','Acceleration',
        'SprintSpeed','Agility','Reactions','Balance','ShotPower',
        'Jumping','Stamina','Strength','LongShots',
        'Aggression','Interceptions','Positioning','Vision',
        'Penalties','Composure','Marking','StandingTackle','SlidingTackle'];

mean = 0

for i in range(0, len(cols)):
    mean += data[cols[i]].mean()
agg_mean = round((mean/2900)*100, 2);
print(agg_mean)

# Labeling data on the basis of whole featureset mean value
# If player's total score is > agg_mean then Above-average Players
# Else 'Below-avergae Players'

label = pd.DataFrame(np.where(round((data.sum(axis =1)/2900)*100,2)
                              > agg_mean, 'Above-average Players',
                              'Below-average Players'))

# Mean normalization of data
data = data.sub(data.mean(axis = 0), axis = 1)

# Converting dataframe to matrix
data_mat = np.asmatrix(data)
data_mat

# calculating covariance now to acheive PCA on the dataset.
# S = (1/n)* XX^T

sigma = np.cov(data_mat.T)
sigma

# Calculating eigen values and eigen vectors
eigVals, eigVec = np.linalg.eig(sigma)

# Sorting eigen values in decreasing order
sorted_index = eigVals.argsort()[::-1]
eigVals = eigVals[sorted_index]
eigVec = eigVec[:,sorted_index]
eigVals
eigVec

# To reduce dimensions of the data set
# from 29 features to 2 features, we select the top 2 eigen vectors

eigVec = eigVec[:,:2]
eigVec

# Transforming data into new sample space
eigVec = pd.DataFrame(np.real(eigVec))
transformed = data_mat.dot(eigVec)
transformed

# Combining the transformed data with its respective labels
final_data = np.hstack((transformed, label))
final_data = pd.DataFrame(final_data)
final_data.columns = ['pc1', 'pc2', 'label']

# PLotting the transformed data
groups = final_data.groupby('label')
figure, axes = plt.subplots()
axes.margins(0.05)
for name, group in groups:
    axes.plot(group.pc1, group.pc2, marker='o', linestyle='', ms=6, label=name)
    axes.set_title("PCA on fifa19 dataset")
axes.legend()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

#Formatting data for K-means cluster analysis
dataK = np.array(list(zip(final_data['pc1'], final_data['pc2'])))
dataK

# Initializing KMeans
kmeans = KMeans(n_clusters=2)
# Fitting with input
kmeans = kmeans.fit(dataK)
# Predicting the clusters
labels = kmeans.predict(dataK)
# Getting the cluster centers
C = kmeans.cluster_centers_
C

#Plotting the clusters
figure, axes = plt.subplots()
axes.margins(0.05)

LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'y'}

label_color = [LABEL_COLOR_MAP[l] for l in labels]
axes.scatter(dataK[:, 0], dataK[:, 1], c=label_color)


axes.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=100)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

#Transforming the labels to numerical values for computing accuracy, classification report and confusion matrix
final_data['label']=np.where(final_data['label']=='Above-average Players', 1, 0)
labels_array=np.array(final_data['label'])
labels_array

#Computing the K-means clusters Accuracy
print("K-means Accuracy:",metrics.accuracy_score(labels_array, labels))
#Computing the error.
print("Mean Absoulte Error:", mean_absolute_error(labels, labels_array))
#Computing classification Report
print("Classification Report:\n", classification_report(labels_array, labels))
#Plotting confusion matrix
print("Confusion Matrix:")
df = pd.DataFrame(
    confusion_matrix(labels_array, labels),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0', '1']])
print(df)


# Players wages prediction using regression

# Distribution of overall rating

bins = np.arange(data['Overall'].min(), data['Overall'].max()+1, 1)
X = data[['Overall']]
plt.figure(figsize=[8,5])
plt.hist(data['Overall'], bins=bins)
plt.title('Overall Rating Distribution')
plt.xlabel('Mean Overall Rating')
plt.ylabel('Count')
plt.show()

# Age vs Overall Rating
plt.figure(figsize=[16,5])
plt.suptitle('Overall Rating Vs Age', fontsize=16)

fig = plt.subplot(1,2,1)
bin_x = np.arange(data['Age'].min(), data['Age'].max()+1, 1)
bin_y = np.arange(data['Overall'].min(), data['Overall'].max()+2, 2)

plt.hist2d(x = data['Age'], y = data['Overall'], cmap="YlGnBu", bins=[bin_x, bin_y])
plt.colorbar()
plt.xlabel('Age (years)')
plt.ylabel('Overall Rating')

plt.subplot(1,2,2)
plt.scatter(x = data['Age'], y = data['Overall'], alpha=0.25, marker='.')

plt.xlabel('Age (years)')
plt.ylabel('Overall Rating')

# Overall Rating vs Potential vs Age
plt.figure(figsize=[8,5])
plt.scatter(x=data['Overall'], y=data['Potential'], c=data['Age'], alpha=0.25, cmap='rainbow' )
plt.colorbar().set_label('Age')
plt.xlabel('Overall Rating')
plt.ylabel('Potential')
plt.suptitle('Overall Rating Vs Potential Vs Age', fontsize=16)
plt.show()

data_pwg = data[['ID','Name', 'Age', 'Overall', 'Potential', 'Value','Wage']]

# overall vs Wage (degree=1)
sns.lmplot(data=data_pwg, x='Overall', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )

# degree = 2
sns.lmplot(data=data_pwg, x='Overall', y='Wage',order=2, scatter_kws={'alpha':0.3, 'color':'y'} )

# degree = 3
sns.lmplot(data=data_pwg, x='Overall', y='Wage',order=3, scatter_kws={'alpha':0.3, 'color':'y'} )

# Wage vs Age(degree =1)
sns.lmplot(data=data_pwg, x='Age', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )

# degree = 2
sns.lmplot(data=data_pwg, x='Age', y='Wage',order=2, scatter_kws={'alpha':0.3, 'color':'y'} )

# degree = 3
sns.lmplot(data=data_pwg, x='Age', y='Wage',order=3, scatter_kws={'alpha':0.3, 'color':'y'} )

# Wage vs Value(degree=1)
sns.lmplot(data=data_pwg, x='Value', y='Wage',order=1, scatter_kws={'alpha':0.3, 'color':'y'} )

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

X = data_pwg[['Age','Overall','Potential','Value']]
y = data_pwg['Wage']

z= data_pwg[['Wage']]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=101)

# Normalizing Train and Test data

from sklearn.preprocessing import StandardScaler
stsc = StandardScaler()
Xtrain = stsc.fit_transform(Xtrain)
Xtest = stsc.fit_transform(Xtest)

def pred_wage(degree, Xtrain, Xtest, ytrain):
    if degree > 1:
        poly = PolynomialFeatures(degree = degree)
        Xtrain = poly.fit_transform(Xtrain)
        Xtest = poly.fit_transform(Xtest)
    lm = LinearRegression()
    lm.fit(Xtrain, ytrain)
    wages = lm.predict(Xtest)
    return wages

# Polynomial regression: Finding the best degree to predict with
MAE, MSE, RMSE = [], [], []

for i in range(1, 11):
    predicted_wages = pred_wage(i, Xtrain, Xtest, ytrain)
    MAE.append(metrics.mean_absolute_error(ytest, predicted_wages))
    MSE.append(metrics.mean_squared_error(ytest, predicted_wages))
    RMSE.append(np.sqrt(metrics.mean_squared_error(ytest, predicted_wages)))

# plotting MAE, MSE, RMSE

plt.figure(figsize=(11,8))
plt.subplot(2,2,1)
plt.plot(MAE, color='red')
plt.xlabel('Degree')
plt.ylabel('Mean Absolute Error')
plt.xlim(-0.5, 10)
plt.ylim(-0.05e7, 0.2e7)
plt.subplot(2,2,2)
plt.plot(MSE, color='green')
plt.xlabel('Degree')
plt.ylabel('Mean Squared Error')
plt.xlim(-0.5, 10)
plt.ylim(-0.25e17, 1e17)
plt.subplot(2,2,3)
plt.plot(RMSE, color='yellow')
plt.xlabel('Degree')
plt.ylabel('Rooted Mean Squared Error')
plt.xlim(-0.5, 10)
plt.ylim(-0.5e8, 2e8)
plt.show()

# As we can see above all the three parameters are minimized at degree = 2 to degree = 6,
# so we can use any degree between them and will apply polynomial regression of degree = 2 to predict wages.

predicted_wages = pred_wage(2, Xtrain, Xtest, ytrain)
predicted_wages

sns.regplot(ytest, predicted_wages, scatter_kws={'alpha':0.3, 'color':'y'})
plt.xlabel('Actual Wage')
plt.ylabel('Predicted Wage')
plt.show()

# Residual plot
sns.distplot(ytest-predicted_wages)
plt.axis([-50000, 50000, 0, 0.00016])

"""
<<<<<<< HEAD
Dveloped by jubin mohanty
"""
=======
AI Course
Jubin Mohanty
"""
>>>>>>> opt
