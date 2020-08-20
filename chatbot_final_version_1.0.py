from telegram.ext import Updater, MessageHandler, Filters
from emoji import emojize
updater = Updater(token='#Enter your token here')
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
# %1 comes after the word remove it from the string and check for the synonmum
chatbot = ChatBot(
    'Terminal',
     storage_adapter='chatterbot.storage.SQLStorageAdapter',
     logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'a',
            'maximum_similarity_threshold': 0.80
        },
        'chatterbot.logic.MathematicalEvaluation',
    ]
)
labels = []
questions = []
number_of_questions=0
for line in open('que.txt', encoding="utf8"):
    labels.append(line.strip().split(" ")[-1])
    questions.append(" ".join(line.strip().split(" ")[:-1]))
    number_of_questions=number_of_questions+1
answers = []
for line in open('ans.txt', encoding="utf8"):
    answers.append(line.strip())
bow_vectorizer = CountVectorizer()

training_vectors = bow_vectorizer.fit_transform(questions)
questions_array=[]
for r in range(0,number_of_questions):
    vector_question=(training_vectors[r].toarray())
    vq=vector_question[0]
    questions_array.append(vq)
#print(questions_array)
labels= np.array(labels)
questions_array=np.array(questions_array)
print(labels)
classifier = GaussianNB()
classifier.fit(questions_array, labels)

def second_algorithm(sentence):
    print(sentence)
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

def handler(bot, update):
  text = update.message.text
  u=text.split(" ")
  chat_id = update.message.chat_id
  response_message=chatbot.get_response(text)
  print(response_message)
  if(str(response_message)=="a"):
    final_response=second_algorithm(text)
  else:
    final_response=response_message
  response_list=str(final_response).split(" ")
  if(len(response_list)>0):
    for r in range(0,len(response_list)):
      if(response_list[r]=="%1"):
        synonyms = []
        input_word=response_list[r-1]
        for syn in wordnet.synsets(input_word): 
          for l in syn.lemmas(): 
              try:
                  w1 = wordnet.synset(input_word+".n.01")
                  w2 = wordnet.synset(l.name()+".n.01")
                  if(w1.wup_similarity(w2)==1):
                      synonyms.append(l.name())
              except:
                  continue
                #  print("An exception occurred")
              if l.antonyms(): 
                  antonyms.append(l.antonyms()[0].name())
        rand=random.randint(0,len(synonyms)-1)
        randomword= synonyms[rand]
        response_list[r-1]=randomword
        response_list[r]=""
        final_response=""
        for m in response_list:
          final_response=final_response+m+" "
        break
  bot.send_message(chat_id=chat_id, text=str(final_response))
  

  
#  if '모해' in text:
  #  bot.send_message(chat_id=chat_id, text='오빠 생각 ㅎㅎ')
 # elif '아잉' in text:
 #   bot.send_message(chat_id=chat_id, text=emojize('아잉:heart_eyes:', use_aliases=True))
 # elif '몇시에' in text:
 #   bot.send_message(chat_id=chat_id, text='7시에 보자')
#  elif '사진' in text:
 #   bot.send_photo(chat_id=chat_id, photo=open('img/mj.jpg', 'rb'))
 # else:
 #   bot.send_message(chat_id=chat_id, text='몰라')

echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)
