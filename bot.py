from tkinter import *
import win32com.client as w
import tensorflow.keras as k
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json,random,pickle

engine1 = w.Dispatch('Sapi.Spvoice')
voices = engine1.GetVoices()
engine1.Voice
engine1.SetVoice(voices.Item(10))

engine2 = w.Dispatch('Sapi.Spvoice')
voices = engine2.GetVoices()
engine2.Voice
engine2.SetVoice(voices.Item(11))

def reply(audio):
    engine1.speak(audio)

def ask(audio):
    engine2.speak(audio)

with open('intents.json') as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()
model = k.models.load_model('chatbot_model.h5')

with open('data.pkl','rb') as f:
    words,labels,training,output = pickle.load(f)

ignore_letters = ['!', '?', ',', '.']

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(w.lower()) for w in s_words if w not in ignore_letters]

    for se in s_words:
        for i,w in enumerate(words):
            if se==w:
                bag[i] = 1

    return np.array([bag])

def chat():    
    tx1=t2.get(1.0, END+"-1c")
    t1.insert(INSERT,f' You : {tx1}\n')
    t2.delete('1.0', END)
    ask(tx1)
    if tx1.lower()=="quit":
        return
    result = model.predict(bag_of_words(tx1,words))
    result_index = np.argmax(result)
    tag = labels[result_index]
    
    for intent in intents['intents']:
        if tag==intent['tag']:
            tx2 = random.choice(intent['responses'])
            t1.insert(INSERT,f' Bot : {tx2}\n\n')
            reply(tx2)
            break            
             
def clear():
    t1.delete('1.0', END)
    t2.delete('1.0', END)
    

root = Tk()
root.geometry('1055x750')
root.maxsize(1055,750)
root.minsize(1055,750)
s = Scrollbar(root)
Label(root,text="My Chatbot",font='Helvetica 18 bold',bg='blue').place(x=500,y=10)
t1 = Text(root,bg='orange',width=92,height=24,font='10',yscrollcommand=s.set)
s.pack(side=RIGHT, fill=Y)
t1.place(x=10,y=50)
s.config(command=t1.yview)
f = Frame(root,width=1015,height=125,bg='blue')
Button(f,text='Send',command=chat,height=3,width=10,font='6').place(x=15,y=20)
Button(f,text='Clear Chat',command=clear,height=3,width=10,font='6').place(x=170,y=20)
t2 = Text(f,bg='orange',width=58,height=4.6,font='10')
t2.place(x=373,y=4)
f.pack(side=BOTTOM,pady=10)
root.config(bg='blue')
if __name__ == "__main__":
    root.mainloop()