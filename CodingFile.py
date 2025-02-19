import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df1=  pd.read_csv(r"C:\Users\LENOVO\Downloads\Abnb_paris - Abnb_paris.csv")
df1.describe()


import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\LENOVO\Downloads\Abnb_paris - Abnb_paris.csv")

# Create an empty DataFrame to store the expanded reviews
expanded_df = pd.DataFrame(columns=df.columns)

# Iterate over each row in the dataset
for index, row in df.iterrows():
    # Split the reviews using the newline character
    reviews = row['Review_text_Act'].split('\n')
    
    # Create a new row for each review and duplicate the other columns
    for review in reviews:
        new_row = row.copy()
        new_row['Review_text_Act'] = review
        expanded_df = expanded_df.append(new_row, ignore_index=True)

# Print the expanded DataFrame
print(expanded_df)


#### Review Cleaning 
import nltk
from nltk.corpus import stopwords
import string
import nltk
from nltk.corpus import stopwords

# Download French stopwords (if not already downloaded)
try:
  stop_words1 = set(stopwords.words('french'))
except LookupError:
  nltk.download('stopwords', language='french')
  stop_words1 = set(stopwords.words('french'))

# Now you have the French stopwords in the 'stop_words' set

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation marks
    text = ''.join(char for char in text if char not in punctuations)
    # Split into words
    words = text.split()
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    words1 = [word for word in words if word not in stop_words1]
    # Join the words back into a string
    cleaned_text = ' '.join(words)
    return cleaned_text
expanded_df['cleaned_text'] = expanded_df['Review_text_Act'].apply(clean_text)
print(expanded_df)
#### Sentiment Score
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# apply sentiment analysis to text column
expanded_df['RTSS'] = expanded_df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
print(df)

#### Calculating the mean sentiment for each property
expanded_df["Unique_id"]=expanded_df.index+1
Sentiment_Score=expanded_df.groupby("Unnamed: 0")["RTSS"].mean()
df1["Unique_id"]=df1.index+1
df1["Sentiment Score"]=Sentiment_Score
##### Final Dataset with Sentiment Values
Abnb_df=  pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\Abnb-Group5.csv")
Abnb_df.describe()
Abnb_df['RTSS'].plot.hist(figsize=(14,6),color='r');
#plt.axvline(x=0.5, color='black', ls='--')
plt.title(f'Sentiment Histogram')
plt.show()

Abnb_df['RTSS'].plot(figsize=(14,6))
plt.ylabel('Daily Mean Sentiment values')
plt.xlabel('Dates')
plt.title(f'Sentiment Trend')
plt.axhline(y=0.5, color='black', ls='--')
plt.show()
####
from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Abnb_df.iloc[:,18:])
    wcss.append(kmeans.inertia_)
km = KMeans(n_clusters=2)
clusters = km.fit_predict(Abnb_df.iloc[:,18:])
Abnb_df["label"] = clusters
print(Abnb_df)
####
df_age = Abnb_df.groupby("cluster")["accommodates"].sum()
print(df_age)

###
# Filter data for monthly fee equal to 1
filtered_df = Abnb_df[Abnb_df['monthfee'] == 1]  # Replace 'monthfee' with your actual column name

# Group by cluster and count occurrences within the filtered data
cluster_value_counts = filtered_df.groupby('cluster')['monthfee'].value_counts().unstack(fill_value=0)

# Plot the cluster value counts as a bar chart
cluster_value_counts.plot(kind='bar')

# Customize labels and title (optional)
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("Distribution of 'monthlyfee' (True) across Clusters (Monthly Fee = 1)")
plt.xticks(rotation=0)  # Optional: Rotate x-axis labels for readability if many clusters
plt.legend(title='monthlyfee')  # Optional: Add legend title

plt.show()


