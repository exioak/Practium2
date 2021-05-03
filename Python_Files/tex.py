import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np 
import networkx as nx 
from flask import Flask


app = Flask(__name__)


def read_article(file_name):
   # f = open(file_name,"r",encoding="utf8")
    
    #filedata = f.readlines()
    article = file_name.split(". ")
    sentences=[]
    x=0
    for s in article:
        sentences.append(s.replace("[^a-zA-Z]"," ").split(" "))
        x=x+1
    sentences.pop()
    #print(x-1)
    return sentences
def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords=[]
    sent1 = [w.lower() for w in sent1]    
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1+sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)]+=1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)]+=1    
    return 1-cosine_distance(vector1,vector2)  
def gen_sim_matrix(sentences,stop_words):
    similarity_matrix=np.zeros((len(sentences),len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1==idx2:
                continue
            similarity_matrix[idx1][idx2]=sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
    return similarity_matrix
def generate_summary(file_name):
    stop_words=stopwords.words('english')
    summarize_text=[]
    sentences = read_article(file_name)
    sentence_similarity_matrix=gen_sim_matrix(sentences,stop_words)
    sentence_similarity_graph=nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s)for i,s in enumerate(sentences)),reverse=True)
    if(len(sentences)>1):
        for i in range(len(sentences)-1):
            summarize_text.append(" ".join(ranked_sentence[i][1]))
    else:
        for i in range(1):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

    #print("Summary \n",". ".join(summarize_text))    
    #print("Summary \n",". \n".join(summarize_text))  
    return ". \n".join(summarize_text)
 
#str = open('a.txt', 'r').read()
#print(str)
#mytext = "India has the second-largest population in the world. India is also knowns as Bharat, Hindustan and sometimes Aryavart. It is surrounded by oceans from three sides which are Bay Of Bengal in the east, the Arabian Sea in the west and Indian oceans in the south. Tiger is the national animal of India. Peacock is the national bird of India. Mango is the national fruit of India. “Jana Gana Mana” is the national anthem of India. “Vande Mataram” is the national song of India. Hockey is the national sport of India. People of different religions such as Hinduism, Buddhism, Jainism, Sikhism, Islam, Christianity and Judaism lives together from ancient times. India is also rich in monuments, tombs, churches, historical buildings, temples, museums, scaenic beauty, wildlife sanctuaries, places of architecture and many more. The great leaders and freedom fighters are from India.Jammu and Kashmir are known as heaven on earth. We can also call Jammu and Kashmir as Tourists Paradise. There are many places to visit Jammu and Kashmir because they have an undisturbed landscape, motorable road, beauty, lying on the banks of river Jhelum, harmony, romance, sceneries, temples and many more. In Jammu and Kashmir, u can enjoy boating, skiing, skating, mountaineering, horse riding, fishing, snowfall, etc. In Jammu and Kashmir, you can see a variety of places such as Srinagar, Vaishnav Devi, Gulmarg, Amarnath, Patnitop, Pahalgam, Sonamarg, Lamayuru, Nubra Valley, Hemis, Sanasar,  Anantnag,  Kargil, Dachigam National Park, Pulwama, Khilanmarg, Dras, Baltal, Bhaderwah, Pangong Lake, Magnetic Hill, Tso Moriri, Khardung La, Aru Valley, Suru Basin,Chadar Trek, Zanskar Valley, Alchi Monastery, Darcha Padum Trek, Kishtwar National Park, Changthang Wildlife Sanctuary, Nyoma, Dha Hanu, Uleytokpo, Yusmarg, Tarsar Marsar Trek and many more. Kerela It is known as the ‘God’s Own Country’, Kerala is a state in India, situated in the southwest region, it is bordered by a number of beaches; covered by hills of Western Ghats and filled with backwaters, it is a tourist destination attracting people by its natural beauty. The most important destinations which you can see in Kerela are the museum, sanctuary, temples, backwaters, and beaches. Munnar, Kovalam, Kumarakom, and Alappad. Conclusion India is a great country having different cultures, castes, creed, religions but still, they live together. India is known for its heritage, spices, and of course, for people who live here. That’s the reasons India is famous for the common saying of “unity in diversity”. India is also well known as the land of spirituality, philosophy, science, and technology."
#generate_summary(mytext,22) 

@app.route("/")
def index():
    return "<h2>API</h2>"
    

@app.route('/<lines>',methods=['POST','GET'])
def summarize(lines):
    summ = generate_summary(str(lines))
    return str(summ)
app.run(debug=True)
    

