# SMS Spam Detection Model

## Project Overview
The SMS Spam Detection Model leverages machine learning and Natural Language Processing (NLP) techniques to classify SMS messages as either spam or legitimate. The project aims to create a robust and scalable system capable of adapting to evolving spam patterns. This ensures enhanced user security and productivity by automating the tedious task of spam detection.

## Features
- **Accurate Classification**: Identifies spam and legitimate messages with high precision and recall.
- **Real-Time Classification**: Provides instant results through a user-friendly interface developed using Streamlit.
- **Adaptability**: Employs retraining mechanisms to stay effective against evolving spam tactics.
- **Metrics for Evaluation**: Uses accuracy, precision, recall, and F1-score to evaluate the model's performance.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, NLTK, Streamlit
- **Machine Learning Algorithms**: Naive Bayes, Support Vector Machines, Ensemble Methods
- **NLP Techniques**: TF-IDF, Word Embeddings

## Methodology
### 1. Data Collection and Preprocessing
- Collected labeled SMS datasets.
- Cleaned, normalized, and tokenized text data to prepare it for analysis.

### 2. Feature Engineering
- Utilized text vectorization techniques like TF-IDF and word embeddings to represent SMS content numerically.

### 3. Model Selection and Training
- Tested multiple machine learning algorithms such as Naive Bayes, Support Vector Machines, and ensemble methods.
- Trained and evaluated models including GaussianNB, BernoulliNB, and MultinomialNB for optimal performance.
  - **GaussianNB**: Achieved an accuracy of 86%.
  - **BernoulliNB**: Achieved an accuracy of 97%.
  - **MultinomialNB**: Achieved an accuracy of 96%.
- Selected the best-performing model based on evaluation metrics.

### 4. Deployment
- Deployed the model using Streamlit to provide a user-friendly interface for real-time SMS classification.

### 5. Monitoring and Retraining
- Incorporated mechanisms for continuous monitoring and retraining to ensure adaptability against new spam patterns.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd sms-spam-detection
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit app in your browser.
2. Input the SMS message you want to classify.
3. View the classification result (Spam or Not Spam) in real-time.

## Results
- The best-performing model achieved high precision and recall, ensuring effective spam detection with minimal false positives.

## Accuracy
- The model achieved an overall accuracy of **95%**, demonstrating its effectiveness in distinguishing spam messages from legitimate ones.

## Future Work
- Integrate deep learning models for more nuanced classification.
- Expand the dataset to improve the model's generalizability.
- Add multilingual support for SMS classification.

## Contributors
- **Aditya Kumar** (Project Lead)
- **Jay Rathord** (Master Trainer)
- AICTE, Microsoft, SAP, and EduEnt Foundation (Project Providers)

## Acknowledgments
Special thanks to the AICTE, Microsoft, SAP, and EduEnt Foundation for providing the project opportunity and resources.

