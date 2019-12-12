import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("pcm_mixed.csv",encoding = 'ISO-8859-1',sep=';')
columns = list(df)
df.replace('Infinity',np.nan,inplace = True)
df.replace(np.nan,1000000,inplace=True)
features=[]
for i in columns:
    if i !='Load':
        features.append(i)
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['Load']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Load']]], axis = 1)

sns.scatterplot(x=finalDf['principal component 1'], y=finalDf['principal component 2'])
# Print the 3d Graph
fig = pyplot.figure()
ax = Axes3D(fig)
sequence_containing_x_vals = finalDf['principal component 1']
sequence_containing_y_vals = finalDf['principal component 2']
sequence_containing_z_vals = finalDf['Load']

ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
ax.set_zlabel('Load')
pyplot.show()

