Sentiment Analysis: A Simple Model for Demonstration
Guide

What is different now is that with ML and generative AI, sentiment analysis can be conducted at scale on any topic at any time, at least in theory. Data scientists with access to sufficient amounts of streaming (live) or historical (regularly updated) data, of any mode (video, text, audio) can now do in a day what used to take professional pollsters weeks and months to assemble. 

This simple model is to demonstrate how ML and generative AI are used to conduct sentiment analysis. This model analyzes one data set made up of historical data of reviews on movies and TV shows in a single mode (digital text) and return the sentiment analysis in two ways: positive or negative. 

1. Tool and Dependencies Setup:
Imported dependencies including the transformers library, huggingface_hub, and the metrics from the datasets library 
Selected the base model: distilBERT, a variation of BERT - distilBERT is BERT's streamlined version, crafted by HuggingFace through knowledge distillation. It mirrors BERT's capabilities but is leaner in terms of parameters

2. Dataset Set Up: 
The IMDB movie review dataset 

3. Tokenization:
Tokenize the dataset via the DistilBERT tokenizer (after dividing the dataset into training and test sets) 

4. Training the Model:
The focus is on classification tasks for sentiment analysis, tapping into DistilBERT's expertise
Used the AutoModelForSequenceClassification from the transformers suite
This was tailored for sequence classification 
Initialized the model
AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
Libraries and API
The Trainer API was used
The TrainingArguments library allows customizing training parameters for improved efficiency
learning rates
batch dimensions
epochs
Using the trainer's .train() method, kickstart the training with cycles of forward and backward passes and optimization

5. Evaluating the Model:
Used the compute_metrics function to analyze evaluation predictions to compute and assess predictions against actual labels
Used the trainer's .evaluate() method to calculate the evaluation data:
Retrieve accuracy score -  load_metric
F1 score - load_metric("f1")
Extracted the logits and labels from eval_pred 
calculate the predictions by selecting the index with the maximum value along the last axis (using np.argmax(logits, axis=-1)).

Resources Used

BERT Documentation (Hugging Face): https://huggingface.co/docs/transformers/model_doc/bert
BERT Official Guide (Hugging Face): https://huggingface.co/docs/transformers/tasks/sequence_classification 
DistilBERT Version, Documentation: https://arxiv.org/abs/1910.01108v4 


References

Deeply Moving: Deep Learning for Sentiment Analysis. “Deeply Moving: Deep Learning for Sentiment Analysis.” Accessed October 13, 2023. http://nlp.stanford.edu/sentiment/index.html.

“Distilbert-Base-Uncased-Finetuned-Sst-2-English · Hugging Face,” June 1, 2023. https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english.

Varshney, Neeraj. “Domain Adaptation for Sentiment Analysis.” Analytics Vidhya (blog), June 2, 2020. https://medium.com/analytics-vidhya/domain-adaptation-for-sentiment-analysis-d1930e6548f4. 
