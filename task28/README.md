# AIML Assignment - Session 28: Emotions Dataset Text Preprocessing Pipeline

## Student Details

- **Name:** Om Prashant Kulawade
- **College:** Zeal Polytechnic, Narhe, Pune
- **Branch:** Artificial Intelligence and Machine Learning (AIML)
- **Subject:** Artificial Intelligence and Machine Learning
- **Session:** 28
- **Topic:** Text Preprocessing using the Emotions Dataset

---

## Objective

The objective of this assignment is to perform text preprocessing on the Emotions Dataset using Python. The preprocessing steps help clean and prepare text data before applying machine learning or deep learning algorithms.

---

## Tools & Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Regular Expressions (re)
- String Module
- NLTK
- Scikit-learn

---

## Dataset

Dataset Used: **Emotions Dataset for NLP**

Files:
- train.txt
- test.txt
- val.txt

Dataset Format:

```
text;emotion
i didnt feel humiliated;sadness
i feel romantic too;love
```

---

## Assignment Tasks

### Q1. Load the Dataset

- Load the dataset using Pandas.
- Display the first 10 rows.
- Display dataset shape.
- Check for missing values.

---

### Q2. Explore Target Labels

- Display unique emotion labels.
- Encode labels using LabelEncoder.
- Create a mapping dictionary.
- Add encoded labels to the DataFrame.

---

### Q3. Lowercase Conversion

- Convert all text to lowercase.
- Compare original and lowercase text.
- Explain why lowercasing is important.

---

### Q4. Remove Punctuation

- Remove punctuation marks.
- Display before and after results.

---

### Q5. Remove Numbers

- Remove digits from the text.
- Compare original and cleaned text.

---

### Q6. Remove Emojis & Special Characters

- Keep only ASCII characters.
- Remove emojis and special symbols.

---

### Q7. Remove Stopwords

- Download NLTK resources.
- Remove English stopwords.
- Display cleaned samples.

---

### Q8. Complete Cleaning Pipeline

Apply all preprocessing steps:

- Lowercase
- Remove punctuation
- Remove numbers
- Remove emojis
- Remove stopwords

Store the cleaned text in a new column called **cleaned_text**.

---

### Q9. Text Length Analysis

- Calculate word count.
- Plot histogram of text lengths.
- Display:
  - Average length
  - Minimum length
  - Maximum length

---

### Q10. Mini Project

Complete preprocessing pipeline:

1. Load dataset
2. Encode labels
3. Clean text
4. Save cleaned dataset as **cleaned_emotions.csv**
5. Display emotion counts

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import string
import re
import nltk
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
```

---

## Output

The assignment performs the following preprocessing tasks:

- Dataset loading
- Label encoding
- Lowercase conversion
- Punctuation removal
- Number removal
- Emoji removal
- Stopword removal
- Complete text cleaning
- Text length analysis
- Save cleaned dataset

---

## Output Files

- `cleaned_emotions.csv`

---

## Conclusion

Text preprocessing is an essential step in Natural Language Processing (NLP). It removes unnecessary information such as punctuation, numbers, emojis, and stopwords while converting text into a standardized format. These preprocessing techniques improve the quality of data and help machine learning models achieve better performance in emotion classification tasks.

---

## Author

**Om Prashant Kulawade**

Zeal Polytechnic, Narhe, Pune

Artificial Intelligence and Machine Learning (AIML)