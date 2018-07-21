# Named-entity-recognition

The goal of this project is learn a named entity recognizer (NER) using the structured perceptron. The named entity recognizer will need to predict for each word one of the following labels:

O: not a named entity;
PER: part of a person's name;
LOC: part of a location's name;
ORG: part of an organization's name;
MISC: part of a name of a different type (miscellaneous).

Implement, train and evaluate a structured perceptron with following features:
current word-current label;
current word-current label and previous label-current label;
as above, but using at least two more features of your choice. Ideas: sub word features, previous/next words, label trigrams, etc.

Please type "python Named-entity-recognition.py train.txt test.txt" in Command Line to execute the Python code.

