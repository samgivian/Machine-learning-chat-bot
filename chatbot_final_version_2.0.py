from telegram.ext import Updater, MessageHandler, Filters
from emoji import emojize
updater = Updater(token='1186645757:AAHDQjRlzdw06wHxc-slGbtXdVMTOVUn8yo')
dispatcher = updater.dispatcher
updater.start_polling()
import json
import random
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
import json
from random import randrange
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from telegram.ext import Updater, InlineQueryHandler, CommandHandler
import requests
import re
from bs4 import BeautifulSoup
from PyDictionary import PyDictionary
import wikipedia

# %1 comes after the word remove it from the string and check for the synonmum
dictionary=PyDictionary("fast")
print(dictionary.printMeanings()) 
print(dictionary.getMeanings()) 
print (dictionary.getSynonyms())
print (dictionary.translate("Hello",'fa'))

chatbot = ChatBot(
    'Terminal',
     storage_adapter='chatterbot.storage.SQLStorageAdapter',
     logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'a',
            'maximum_similarity_threshold': 0.95
        },
        'chatterbot.logic.MathematicalEvaluation'
    ]
)


labels = []
questions = []
number_of_questions=0
answers = []
questions_array=[]
emoji_array=[]
def emoji_list():
    emoji=[]
    for line in open('emoji.txt', encoding="utf8"):
        emoji.append((line.strip()+" emoji"))
        
    return emoji

emoji_array=emoji_list()
#print(emoji_array)

def reading_questions(number_of_questions):
    n=number_of_questions
    for line in open('que.txt', encoding="utf8"):
        labels.append(line.strip().split(" ")[-1])
        questions.append(" ".join(line.strip().split(" ")[:-1]))
        n=n+1
    for i in emoji_array :
        labels.append(str(int(labels[len(labels)-1])+1))
        p="send " +i
        questions.append(" ".join(p.strip().split(" ")))
        n=n+1
    return n

def reading_answers():
    for line in open('ans.txt', encoding="utf8"):
        answers.append(line.strip())
    for i in emoji_array:
        p=i
        answers.append(p.strip())
        

number_of_questions=reading_questions(number_of_questions)
reading_answers()
#print(questions)
#print(answers)
#print(labels)
bow_vectorizer = CountVectorizer()
training_vectors = bow_vectorizer.fit_transform(questions)

for r in range(0,number_of_questions):
    vector_question=(training_vectors[r].toarray())
    vq=vector_question[0]
    questions_array.append(vq)


labels= np.array(labels)
questions_array=np.array(questions_array)
#print(labels)
classifier = GaussianNB()
classifier.fit(questions_array, labels)



#getting emojies

def second_algorithm(sentence):
   # print(sentence)
    input_vector = bow_vectorizer.transform([sentence]).toarray()
    print(input_vector)
    predict = classifier.predict(input_vector)
    print(predict)
    index = int(predict[0])
    print("Accurate:",str(classifier.predict_proba(input_vector)[0][index-1] * 100)[:5] + "%")
    answers_list=[x.strip() for x in answers[index-1].split(',')]
    rand=random.randint(0,len(answers_list)-1)
    answer=answers_list[rand]
    return answer


def get_url():
    contents = requests.get('https://random.dog/woof.json').json()    
    url = contents['url']
    return url


def get_image_url():
    allowed_extension = ['jpg','jpeg','png']
    file_extension = ''
    while file_extension not in allowed_extension:
        url = get_url()
        file_extension = re.search("([^.]*)$",url).group(1).lower()
    return url


def bop(bot, update):
    url = get_image_url()
    chat_id = update.message.chat_id
    bot.send_photo(chat_id=chat_id, photo=url)

print("Start")
def handler(bot, update):

    #reading the text by user
    text = update.message.text
    u=text.split(" ")
    chat_id = update.message.chat_id
    final_response=chatbot.get_response(text)
    #print(response_message)

    #calling the word matching algorithm 
    if(str(final_response)=="a"):
        final_response=second_algorithm(text)
        if(final_response=="my pic"):
            bot.send_photo(chat_id=chat_id, photo=open('chatbotpic.jpg', 'rb'))

        #sending emoji
        #print(final_response)
        #print(emoji_array)
        elif(final_response in emoji_array):
            bot.send_message(chat_id=chat_id, text=emojize(':'+final_response.split(" ")[0]+':', use_aliases=True))

        elif(final_response == "dog pic"):
            url = get_image_url()
            chat_id = update.message.chat_id
            bot.send_photo(chat_id=chat_id, photo=url)

        elif(final_response=="wiki"):
            word_to_search=""
            for m in range(1,len(u)):
                word_to_search=word_to_search+u[m]
          
            print(word_to_search)
            result = wikipedia.summary(word_to_search, sentences=2)
            bot.send_message(chat_id=chat_id, text=str(result))
            
        else:
            bot.send_message(chat_id=chat_id, text=str(final_response))
    else:
        bot.send_message(chat_id=chat_id, text=str(final_response))


echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)
