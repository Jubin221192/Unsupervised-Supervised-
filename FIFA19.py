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

# Finding the null values:
null_val = data.isna().sum()
print(data.shape)

# Data visualization for understanding the data

# Comparison of preferred foot over the different p layers
data['Preferred Foot'].value_counts().head(50).plot.bar(color= 'purple')

# Comparison of international rep
data['International Reputation'].value_counts()

# plotting a pie chart to represent share of international reputation
labels = ['1', '2', '3', '4', '5']
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
ax.set_xlabel(xlabel= "Player\ 's Potential Scores", fontsize = 16)
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
axes.scatter(dataK[:, 0], dataK[:, 1], c=labels)
axes.scatter(C[:, 0], C[:, 1], marker='*', c='#060502', s=100)
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






