import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
import random,pickle
import tensorflow.keras as k
# nltk.download('punkt')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open('intents.json') as f:
    intents = json.load(f)

words=[]
labels = []
docs_x = []
docs_y = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)   # tokenize each word and returns a list 
        words.extend(word)
        docs_x.append(word)
        docs_y.append(intent['tag'])
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]

words = sorted(list(set(words)))    # total vocabulory
labels = sorted(labels)

training = []
output = []
output_empty = [0 for _ in range(len(labels))]

for x,doc in enumerate(docs_x):
    bag = []

    wrds = [lemmatizer.lemmatize(w.lower()) for w in doc if w not in ignore_letters] 
    
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)    
    
    output_row = output_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

with open('data.pkl','wb') as f:
    pickle.dump((words,labels,training,output),f)

# Model

model = k.Sequential()
model.add(k.layers.Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(k.layers.Dropout(0.5))
model.add(k.layers.Dense(64, activation='relu'))
model.add(k.layers.Dropout(0.5))
model.add(k.layers.Dense(len(output[0]), activation='softmax'))

sgd = k.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(training,output, epochs=800, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")