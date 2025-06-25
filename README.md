🧠 Mental Health Risk Prediction Using Personality Traits and Text Analysis This project aims to predict mental health risk levels (Low, Mid, High) based on a user's text input, leveraging personality trait inference and deep learning techniques. The pipeline is designed to assist early detection of mental health concerns using natural language processing (NLP) and psychological profiling.

🔍 Key Features: DistilBERT-based Personality Prediction: A fine-tuned regression model that predicts continuous values of the Big Five personality traits (OCEAN) from user-generated text.

GRU-based Mental Health Classifier: A lightweight and efficient GRU model (ConcatPoolingGRUAdaptive) that classifies users into Low, Mid, or High risk levels based on the predicted personality scores.

End-to-End Pipeline: The system takes raw text as input, extracts personality insights, scales them, and classifies mental health risk—all in one streamlined flow.

Datasets Used:

MBTI Dataset for personality modeling.

CLPsych Dataset and custom-annotated text data for mental health classification.

Tech Stack: Python, PyTorch, Hugging Face Transformers, Pandas, Scikit-learn

🎯 Objectives: Explore the intersection of NLP and mental health.

Develop an interpretable and practical AI tool for risk identification.

Provide a scalable framework for academic or clinical research use.
