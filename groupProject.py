# Step 1: Load and Preview the Dataset
import pandas as pd
print("Loading the dataset...")

# Load dataset
file_path = 'Data4.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
data.info()
basic_info = {
    "Total Records": len(data),
    "Columns": list(data.columns),
    "Missing Values": data.isnull().sum().to_dict(),
    "Data Types": data.dtypes.to_dict()
}
print("\nBasic Info:", basic_info)

# Preview the dataset
data_preview = data.head()
print("\nData Preview:")
print(data_preview)

# Step 2: Vectorize Text Data
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords from NLTK
nltk.download('stopwords')
print("\nVectorizing the text data...")

# Initialize CountVectorizer with stopwords
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))

# Transform the 'Body' column using CountVectorizer
X_counts = vectorizer.fit_transform(data['Body'])
vocabulary_size = len(vectorizer.get_feature_names_out())
transformed_shape = X_counts.shape

# Display transformation information
transformation_info = {
    "Vocabulary Size": vocabulary_size,
    "Transformed Data Shape": transformed_shape
}
print("\nCount Vectorization Info:", transformation_info)

# Highlight information for initial features
highlight_info = {
    "Number of Records": X_counts.shape[0],
    "Number of Features": X_counts.shape[1],
    "Vocabulary Size": len(vectorizer.get_feature_names_out()),
    "Sample Features": vectorizer.get_feature_names_out()[:10]  # First 10 features
}
print("\nHighlight Info:", highlight_info)

# Step 3: Apply TF-IDF Transformation
from sklearn.feature_extraction.text import TfidfTransformer
print("\nApplying TF-IDF Transformation...")

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Display TF-IDF transformation highlights
tfidf_highlight_info = {
    "Number of Records": X_tfidf.shape[0],
    "Number of Features": X_tfidf.shape[1],
    "Sample TF-IDF Values": X_tfidf[0, :10].toarray().flatten().tolist(),  # First 10 values
    "Sample Features": vectorizer.get_feature_names_out()[:10]  # First 10 features
}
print("\nTF-IDF Transformation Info:", tfidf_highlight_info)

# Step 4: Shuffle and Split the Dataset
print("\nShuffling and Splitting the dataset...")
shuffled_data = data.sample(frac=2, replace=True, random_state=42).reset_index(drop=True)
X = shuffled_data['Body']
y = shuffled_data['Label']
split_index = int(len(shuffled_data) * 0.8)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
split_info = {
    "Total Records (Shuffled and Upsampled)": len(shuffled_data),
    "Training Set Size": len(X_train),
    "Testing Set Size": len(X_test)
}
print("\nDataset Split Info:", split_info)

# Step 5: Train and Evaluate Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
print("\nTraining Naive Bayes Classifier...")

# Transform text data for training and testing
X_train_counts = vectorizer.transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)
print("\nNaive Bayes Classifier trained.")

# Perform 3-fold cross-validation
cv_scores = cross_val_score(nb_classifier, X_train_counts, y_train, cv=3, scoring='accuracy')
mean_cv_accuracy = cv_scores.mean()
print("\nMean Cross-Validation Accuracy:", mean_cv_accuracy)

# Test the model and compute metrics
y_test_pred = nb_classifier.predict(X_test_counts)
conf_matrix = confusion_matrix(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("\nTest Results:")
print("Confusion Matrix:")
print(conf_matrix)
print("Test Accuracy:", test_accuracy)