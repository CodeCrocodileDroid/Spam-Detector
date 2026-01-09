# Spam-Detector
Python Machine Learning and NLP
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/56ea290f-1af6-4a4c-b20d-354efd8bcb7b" />

📧 Spam Email Classifier
A machine learning system for classifying emails as Spam (unwanted/junk) or Ham (legitimate/wanted) using Natural Language Processing (NLP) and multiple classification algorithms.

🚀 Features
Multiple ML Models: Naive Bayes, Logistic Regression, SVM, and Random Forest

NLP Pipeline: Text preprocessing with tokenization, lemmatization, and stopword removal

Model Persistence: Save/load trained models for production use

Performance Analysis: Comprehensive evaluation metrics and model comparison

Interactive Testing: Command-line interface for testing new emails


Ham and Spam are terms used to categorize emails in spam filtering:

Spam
Definition: Unwanted, unsolicited emails sent in bulk

Characteristics:

Commercial advertisements (especially for pharmaceuticals, loans, adult content)

Phishing attempts

Scams and fraudulent offers

Mass marketing emails you didn't subscribe to

Typically sent to large numbers of recipients

Examples from your dataset:

Viagra/Cialis advertisements

Stock tips and investment opportunities

Fake lottery winnings

Nigerian prince/419 scams

Pornography/dating site promotions

Ham
Definition: Legitimate, wanted emails

Characteristics:

Personal emails from friends/family

Work/business communications

Newsletters you subscribed to

Purchase receipts

Important notifications

Non-commercial messages

Examples:

"Meeting at 3 PM tomorrow"

"Your Amazon order has shipped"

"Photos from our vacation"

"Resume attached as requested"

Key Differences:
Aspect	Ham (Legitimate)	Spam (Junk)
Consent	Recipient opted in or expects it	Unsolicited
Relevance	Usually relevant to recipient	Often irrelevant
Volume	Sent to individuals/small groups	Mass distribution
Intent	Communication/information	Commercial gain/scam
Quality	Generally well-written	Often poorly written
Personalization	Often personalized	Generic/impersonal
In Your Dataset:
Looking at your CSV file, the spam column uses:

1 = Spam (the emails you provided are all spam)

0 = Ham (not present in your sample)

Historical Context:
"Spam" comes from a Monty Python sketch where Spam (the canned meat) is repeated excessively

"Ham" is the opposite - legitimate communication (like "ham radio" operators)

The terms were popularized by early email filters like SpamAssassin

In Machine Learning:
Your classifier learns patterns to distinguish between:

Spam indicators: Words like "Viagra", "free", "click here", "$$$", "guaranteed", etc.

Ham indicators: Normal language, personal names, specific details, professional tone

The goal is to correctly identify spam (catching junk emails) while minimizing false positives (not misclassifying legitimate emails as spam).
