import os
import zipfile
import os

zip_path = 'C:\\Users\\NISHITA\\Downloads\\final_model\\Transcripts.zip'
extract_to = 'C:\\Users\\NISHITA\\Downloads\\final_model\\Transcripts'

# Unzip only if not already extracted
if not os.path.exists(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Now point your path to the extracted folder
transcript_path = extract_to

# List all CSV files
csv_files = [f for f in os.listdir(transcript_path) if f.endswith('.csv')]


transcript_path = 'C:\\Users\\NISHITA\\Downloads\\final_model\\Transcripts\\Transcripts'

# List all CSV files
csv_files = [f for f in os.listdir(transcript_path) if f.endswith('.csv')]

# Example: Read one file using pandas
import pandas as pd
edf = pd.read_csv(os.path.join(transcript_path, csv_files[0]))
print(edf.head())


label_path='C:\\Users\\NISHITA\\Downloads\\final_model\\Detailed Labels.csv'
df1 = pd.read_csv(label_path)
print(df1.head())

train_df = df1
print(train_df.head())

Train_ids = train_df['Participant'].astype(str).tolist()

features = []
import pandas as pd
import os
import re

features = []

for file in csv_files:
    pid = file.split('_')[0]  # Extract ID from filename
    if pid not in Train_ids:
        continue

    df = pd.read_csv(os.path.join(transcript_path, file))

    total_duration = df['End_Time'].max() - df['Start_Time'].min()
    total_utterances = len(df)
    avg_confidence = df['Confidence'].mean()
    total_words = df['Text'].astype(str).str.split().map(len).sum()
    avg_words_per_utterance = total_words / total_utterances

    # Compute pause-related features
    start_times = df['Start_Time'].values
    end_times = df['End_Time'].values
    pauses = start_times[1:] - end_times[:-1]
    avg_pause_duration = pauses.mean() if len(pauses) > 0 else 0
    max_pause_duration = pauses.max() if len(pauses) > 0 else 0

    # Full concatenated text
    full_text = " ".join(df['Text'].astype(str)).lower()
    words = re.findall(r'\b\w+\b', full_text)
    unique_words = len(set(words))
    lexical_diversity = unique_words / total_words if total_words else 0
    avg_word_len = sum(len(w) for w in words) / total_words if total_words else 0

    # Sleep-related keywords
    sleep_keywords = ['sleep', 'tired', 'insomnia', 'rest', 'dream', 'awake', 'nap', 'fatigue']
    sleep_word_count = sum(full_text.count(word) for word in sleep_keywords)

    # Pronoun and negation features
    first_person_pronouns = sum(full_text.count(w) for w in ['i', 'me', 'my'])
    negations = sum(full_text.count(w) for w in ['not', "don’t", "can't", "won’t", "no", "never"])

    features.append({
        'Participant_ID': pid,
        'Total_Duration': total_duration,
        'Total_Utterances': total_utterances,
        'Avg_Confidence': avg_confidence,
        'Total_Words': total_words,
        'Avg_Words_Per_Utterance': avg_words_per_utterance,
        'Avg_Pause_Duration': avg_pause_duration,
        'Max_Pause_Duration': max_pause_duration,
        'Lexical_Diversity': lexical_diversity,
        'Avg_Word_Length': avg_word_len,
        'Sleep_Word_Count': sleep_word_count,
        'First_Person_Pronouns': first_person_pronouns,
        'Negation_Count': negations
    })

features_df = pd.DataFrame(features)
print(features_df.head())
from sklearn.preprocessing import MinMaxScaler

# Select columns to normalize (excluding ID and label)
columns_to_normalize = ['Total_Duration', 'Total_Utterances', 'Avg_Confidence',
                        'Total_Words', 'Avg_Words_Per_Utterance', 'Avg_Pause_Duration',
                        'Max_Pause_Duration', 'Lexical_Diversity', 'Avg_Word_Length',
                        'Sleep_Word_Count', 'First_Person_Pronouns', 'Negation_Count']

# Normalize
scaler = MinMaxScaler()
features_df[columns_to_normalize] = scaler.fit_transform(features_df[columns_to_normalize])

print(features_df.head())

from sklearn.feature_extraction.text import CountVectorizer

# Collect all participant transcripts
participant_texts = []

participant_ids = []
for file in csv_files:
    pid = file.split('_')[0]
    if pid not in Train_ids:
        continue

    df = pd.read_csv(os.path.join(transcript_path, file))
    full_text = ' '.join(df['Text'].dropna().astype(str).tolist())
    participant_texts.append(full_text)
    participant_ids.append(pid)

# Create binary Bag of Words model
vectorizer = CountVectorizer(binary=True, max_features=1000)  # limit vocab if needed
X_bow = vectorizer.fit_transform(participant_texts).toarray()

# Create DataFrame
bow_df = pd.DataFrame(X_bow, columns=[f'word_{w}' for w in vectorizer.get_feature_names_out()])
bow_df['Participant_ID'] = participant_ids

# Merge with features_df
#features_df = features_df.merge(bow_df, on='Participant_ID')

final_df = bow_df.merge(features_df, on='Participant_ID', how='inner')

final_df_1 = bow_df.merge(features_df, on='Participant_ID', how='inner')


X = final_df.drop(['Participant_ID'], axis=1)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# Create new DataFrame with PCA components
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df['Participant_ID'] = final_df['Participant_ID'].values

print(pca_df.head())

import numpy as np
train_df
train_df['Sleep_Disorder'] = np.where((train_df['PHQ8_3_Sleep'] >= 2) & (train_df['PCL-C_13_Sleep'] >= 3), 1, 0)
train_df.loc[(train_df['PHQ8_3_Sleep'] == 3) | (train_df['PCL-C_13_Sleep'] >= 4), 'Sleep_Disorder'] = 1

final_df=pca_df
train_df = train_df.loc[:, ~train_df.columns.duplicated()]


# Remove duplicate column names if any
train_df = train_df.rename(columns={'Participant': 'Participant_ID'})
train_df = train_df.loc[:, ~train_df.columns.duplicated()]

# Ensure both Participant_ID columns are strings
final_df['Participant_ID'] = final_df['Participant_ID'].astype(str)
train_df['Participant_ID'] = train_df['Participant_ID'].astype(str)

# Merge
# Remove duplicate column names
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

# Now safely subset and merge
final_df = final_df.merge(train_df[['Participant_ID', 'PHQ8_3_Sleep']], on='Participant_ID', how='inner')

final_df = final_df.merge(train_df[['Participant_ID', 'split','PTSD_severity','Depression_severity','Sleep_Disorder']], on='Participant_ID', how='inner')


test_df = final_df[final_df['split'] == 'test']
train = final_df[final_df['split'].isin(['train', 'dev'])]

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

X_train = train.drop(['Participant_ID', 'PHQ8_3_Sleep', 'split','Sleep_Disorder','Depression_severity'], axis=1)
X_test = test_df.drop(['Participant_ID', 'PHQ8_3_Sleep', 'split','Sleep_Disorder','Depression_severity'], axis=1)
y_train = train['Sleep_Disorder']
y_test = test_df['Sleep_Disorder']
participant_ids_test = test_df["Participant_ID"].values

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Train text model
model_text = DecisionTreeClassifier(max_depth=3)
model_text.fit(X_train, y_train)

# Predict labels and probabilities
y_pred_text = model_text.predict(X_test)
y_proba_text = model_text.predict_proba(X_test)  # shape: (n_samples, n_classes)

# Print metrics
print(model_text.get_depth())
print(classification_report(y_test, y_pred_text))
print(confusion_matrix(y_test, y_pred_text))

# Construct DataFrame to save probabilities
text_proba_df = pd.DataFrame({
    'Participant_ID': test_df['Participant_ID'].values,  # Ensure this matches X_test order
    'Text_Prob_0': y_proba_text[:, 0],
    'Text_Prob_1': y_proba_text[:, 1],
})

# Save to Excel
text_proba_df.to_excel('text_probabilities.xlsx', index=False)