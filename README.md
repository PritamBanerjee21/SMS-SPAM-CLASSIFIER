# SMS Spam Classifier Web App

[Dataset Link] : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

[Live Web App] : https://pritam-sms-spam-classifier.streamlit.app/

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Data Cleaning](#data-cleaning)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Vectorization & Data Splitting](#vectorization--data-splitting)
- [Model Building](#model-building)
- [Deployment](#deployment)
- [Conclusion](#conclusion)
- [How to Use the Code](#how-to-use-the-code)

## Project Overview
This project involves creating a web application for classifying SMS messages as either 'ham' (not spam) or 'spam'. The application is built using Streamlit and deployed on Streamlit Cloud. The model uses various machine learning algorithms, with a focus on precision to minimize the risk of false positives in spam detection. Also I have used 'Functional Programming' to increase the readability and reusability of the code.

## Dataset Description
The dataset used in this project was sourced from Kaggle and initially contained five columns: `v1`, `v2`, `unnamed:2`, `unnamed:3`, and `unnamed:4`. The dataset was imbalanced and required significant cleaning and preprocessing to be usable for model training.

## Data Cleaning
- **Handling Missing Data**: The last three columns had more than 90% missing values and were removed.
- **Renaming Columns**: The columns `v1` and `v2` were renamed to `TARGET` and `TEXT`, respectively.
- The resulting cleaned dataset is referred to as `CLEAN_DF`.

## Preprocessing
- **Text Normalization**: All text data was converted to lowercase.
- **Stopwords & Punctuation Removal**: Common stopwords and punctuation marks were removed.
- **Character Filtering**: Only alphanumeric characters were retained.
- The preprocessed data is stored in a dataframe named `PREP_DF`.

## Feature Extraction
Three additional features were extracted from the `CLEAN_DF`:
1. `num_char`: Number of characters in the message.
2. `num_words`: Number of words in the message.
3. `num_sent`: Number of sentences in the message.
These features were stored in a new dataframe named `NEW_DF`.

## Exploratory Data Analysis (EDA)
- **Distribution Visualization**: The distribution of characters, words, and sentences was visualized for both 'ham' and 'spam' messages.
- **Top Words Visualization**: The top 30 words in 'spam' and 'ham' messages were displayed using bar plots.

## Vectorization & Data Splitting
- **Vectorization**: The `TEXT` column was vectorized using `TfidfVectorizer` with max_features = 3000.
- **Feature Scaling**: The additional features in `NEW_DF` were scaled using `MinMaxScaler`.
- **Data Splitting**: Both `PREP_DF` and `NEW_DF` were split into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`).

## Model Building
Six models were built and evaluated:
1. **GaussianNB**
2. **MultinomialNB**
3. **BernoulliNB**
4. **Logistic Regression**
5. **Random Forest Classifier**
6. **XGBoost Classifier**

The models were trained and tested on both `PREP_DF` and `NEW_DF` to determine the impact of the extracted features. The `BernoulliNB` model demonstrated the highest precision, making it the preferred model for deployment. The precision and accuracy were identical across both datasets, leading to the decision to deploy the model without the additional features.

Accuracy : 98%

Precision Score : 100%

## Deployment
The final model (`BernoulliNB`) and the `TfidfVectorizer` were exported using `joblib`. The web app was then developed using Streamlit and deployed on Streamlit Cloud for public access.

## Conclusion
This project successfully demonstrates the process of building a reliable SMS spam classifier. Emphasizing precision over accuracy ensures a robust model that effectively minimizes false positives, which is crucial in spam detection.

## How to Use the Code
1. **Clone the Repository**: Use `git clone` to download the repository.
2. **Install Dependencies**: Run `pip install -r requirements.txt` to install the necessary libraries.
3. **Run the App**: Use `streamlit run app.py` to start the web application locally.
4. **Explore the Code**: The code is structured into various modules for ease of understanding and modification.

