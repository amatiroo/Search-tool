import os
import math
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stop_tokens = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
doc_freq = {}
doc_freq_unique = Counter()
Posting_list = {}
corpusroot = '/home/roo/Documents/Data/P1/P1/presidential_debates/presidential_debates/' #my directory path where data files are stored 
doc_tokens = {}
num_of_docs=0
for filename in os.listdir(corpusroot): # to read each file from directory
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    
    doc = file.read()  # store the file content in variable doc
    doc = doc.lower()   # convert all the words in doc to lower case
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+') 
    tokens = tokenizer.tokenize(doc) # tokenize the content of doc and store it in tokens variable
    doc1 = []   
    for w in tokens:        # for loop to stem the tokenized words in doc
        if w not in stop_tokens:
            x=stemmer.stem(w)
            doc1.append(x)
    
    term_freq_doc = Counter(doc1) # counts the frequency of each term in doc1 
    
    
    doc_freq_unique += Counter(list(set(term_freq_doc))) # list unique words in the whole corpus and their document frequency
    
    doc_tokens[filename] = term_freq_doc # storing term frequency of words for each doc
    num_of_docs+=1
    
    
    file.close()



#calculating  weighted inverse df using formula
for i in doc_freq_unique:
    doc_freq[i] = math.log(num_of_docs/doc_freq_unique[i],10)
    
    


#calculating weighted term frequency using formula

for file in doc_tokens:
    item = doc_tokens[file]
    sum_of_tf = 0
    for i in item:
        
        tfw = (1+math.log(doc_tokens[file][i],10))
       
        doc_tokens[file][i]= tfw
        
    


    

#function to get weighted term frequency of a term    
def get_tf(filename,term):
    
    if filename in doc_tokens:
        x = doc_tokens[filename]
        if term in x:
            return (doc_tokens[filename][term])
        else:
            return 0
    else:
        return "invalid filename"
    

#function to get weighted inverse df of a term   
def getidf(term):
    if term in doc_freq:
        return doc_freq[term]
    else:
        return -1
    
    
    
tfidf = {}  
magnitude = {}
sum_of_tfidf =0
# In below for loop we calculate tfidf scores for each term in doc , and also fill values the Posting list


for file in doc_tokens:
    x = doc_tokens[file]
    tfidf[file] = {}
    sum_of_tfidf =0
    for item in x:
        tf = get_tf(file,item)
        idf = getidf(item)
        
            
        tfidf[file][item]=tf*idf
        
        sum_of_tfidf += tfidf[file][item]**2
        if item not in Posting_list:
            Posting_list[item]=Counter()
        Posting_list[item][file]=tfidf[file][item]
            
        
    magnitude[file] = math.sqrt(sum_of_tfidf)
    
                   
#below function is to get the normalized tfidf scores
def getweight(filename,term):
    if term in tfidf[filename]:
        return (tfidf[filename][term]/magnitude[filename])
    else:
        return 0
    
    
#In below for loop we fill the normalized value of tfidf scores in Posting list 
for file in doc_tokens:
    x = doc_tokens[file]
    
    for item in x:
        Posting_list[item][file]=getweight(file, item)
 
#below function is to get the query score 
def query(qstring):
    common_docs = []
    most_common_doc = {}
    qstring.lower()
    ten = {}
    flag=0
    
    cos_sim_docs=Counter() 
    qstring_tf = {}
    qstring=qstring.split(' ')
    
    
    
    sum_of_qtf = 0
    x=0
    # below for loop does below things
    # calculate tf of query string
    #get the top 10 docs from posting list if query term exists , else gets random common docs
    #get the all the common docs in which query terms exists 
    for x in qstring:
        term = stemmer.stem(x)
        if term not in Posting_list:  
            continue
        
        termcount = qstring.count(term)
        if termcount == 0:
            qstring_tf[term] = 1
        else:
            qstring_tf[term] = (1+math.log(termcount,10))
        
        sum_of_qtf += (qstring_tf[term]**2)
        
        if getidf(term)==-1:                   
            most_common_doc[term], weights = zip(*Posting_list[term].most_common())       
        else:
            most_common_doc[term],weights = zip(*Posting_list[term].most_common(10))    
        ten[term]=weights[9]            
           
        if flag==1:
            common_docs=set(most_common_doc[term]) & common_docs    
            
        else:
            common_docs=set(most_common_doc[term])
            flag=1
            
    magnitude_of_qtf = math.sqrt(sum_of_qtf)   
    #below for loop calculates the cosine similarity
    for doc in tfidf:
        cos_sim=0
        for term in qstring_tf:
            if doc in most_common_doc[term]:
                cos_sim = cos_sim + (qstring_tf[term]  / magnitude_of_qtf) * Posting_list[term][doc]
                
            else:
                cos_sim = cos_sim + (qstring_tf[term]  / magnitude_of_qtf) * ten[term]  

        cos_sim_docs[doc]=cos_sim
   
    max=cos_sim_docs.most_common(1) 
    result_doc,score=zip(*max)
    
    
    if common_docs:
        if result_doc[0] in common_docs:
            return result_doc[0],score[0]
        else:
            return "fetch more",0   
    else:
        return "None",0   
        
        

print("%.12f" % getidf("health"))
print("%.12f" % getidf("agenda"))
print("%.12f" % getidf("vector"))
print("%.12f" % getidf("reason"))
print("%.12f" % getidf("hispan"))
print("%.12f" % getidf("hispanic"))
print("%.12f" % getweight("2012-10-03.txt","health"))
print("%.12f" % getweight("1960-10-21.txt","reason"))
print("%.12f" % getweight("1976-10-22.txt","agenda"))
print("%.12f" % getweight("2012-10-16.txt","hispan"))
print("%.12f" % getweight("2012-10-16.txt","hispanic"))
print("(%s, %.12f)" % query("health insurance wall street"))
print("(%s, %.12f)" % query("particular constitutional amendment"))
print("(%s, %.12f)" % query("terror attack"))
print("(%s, %.12f)" % query("vector entropy"))
    
    
    
    
    
    
    
        
    
      