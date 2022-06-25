import pandas as pd 
import re 
import spacy 
from torchtext.vocab import FastText
import torch 
import numpy as np
from torch.utils.data import DataLoader, Dataset





data = pd.read_csv(r'C:\Users\ritth\code\Strive\Strive-Exercises\Chapter 03\15. Toxicity detect\train.csv\train.csv')
# print(data.head())

data = data.drop('id', axis=1)
# print(data)



def data_engineering(data):
     column_name, no, yes = [], [], []
     for i in data.columns:
          if i == 'comment_text':
               continue
          else:
               column_name.append(i),   no.append(data[i].value_counts()[0]),   yes.append(data[i].value_counts()[1])
     return column_name, no, yes

# column_name, no, yes = data_engineering(data)

# print(column_name)
# print(no)
# print(yes)

# print('BEFORE')
# print(data['comment_text'][3])



def preprocessing(data):
    
     data['comment_text'] = data['comment_text'].apply(lambda text: text.lower()) 
     data['comment_text'] = data['comment_text'].apply(lambda text: text.strip()) 
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('\n', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub(',', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('.', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('\'', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('"', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('! ', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('/', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('-', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('\w*\d\w*\*', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub(r'[^\x00-\x7f]', r' ', text)) 
     data['comment_text'] = data['comment_text'].apply(lambda text: text.strip())
     return data 


data = preprocessing(data)
# print(data['comment_text'][3])




def split_data(data):
     data_toxic          = data[['comment_text', 'toxic']]
     data_severe_toxic   = data[['comment_text', 'severe_toxic']]
     data_obscene        = data[['comment_text', 'obscene']]
     data_threat         = data[['comment_text', 'threat']]
     data_insult         = data[['comment_text', 'insult']]
     data_identity_hate  = data[['comment_text', 'identity_hate']]
     return data_toxic, data_severe_toxic, data_obscene, data_threat, data_insult, data_identity_hate 


data_toxic, data_severe_toxic, data_obscene, data_threat, data_insult, data_identity_hate = split_data(data) 

# print()
# print(data_toxic.head(5))
# print()
# print(data_severe_toxic.head(5))
# print()
# print(data_obscene.head(5))
# print()
# print(data_threat.head(5))
# print()
# print(data_insult.head(5))
# print()
# print(data_identity_hate.head(5))





nlp = spacy.load("en_core_web_sm")


# split
def train_test_split(df, train_size=0.7):

    df_idx = [i for i in range(len(df))]
    np.random.shuffle(df_idx)

    len_train = int(len(df) * train_size)
    train_idx, test_idx = df_idx[:len_train], df_idx[len_train:]

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)





def preprocessing(sentence):
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return tokens


#data_toxic_few = data_toxic.iloc[:100]

# data_toxic_few['comment_text'] = data_toxic_few['comment_text'].apply(lambda text: preprocessing(text))
# print(data_toxic_few)



def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0

def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]


def front_padding(list_of_indexes, max_seq_len, padding_index=1):
    new_out = (max_seq_len - len(list_of_indexes))*[padding_index] + list_of_indexes
    return new_out[:max_seq_len]  



fasttext = FastText("simple")

#example = front_padding(encoder(preprocessing("This is VERY toxic review coming from you"), fasttext), max_seq_len=max_sentence)
# print(example) 


class Data(Dataset):
    def __init__(self, data,data_target, max_seq_len=32): # data is the input data, max_seq_len is the max lenght allowed to a sentence before cutting or padding
        self.max_seq_len = max_seq_len
        
        
        train_iter = iter(data['comment_text'].values)
        self.vec = FastText("simple")

        self.vec.vectors[1] = -torch.ones(self.vec.vectors[1].shape[0]) # replacing the vector associated with 1 (padded value) to become a vector of -1.
        self.vec.vectors[0] = torch.zeros(self.vec.vectors[0].shape[0]) # replacing the vector associated with 0 (unknown) to become zeros
        self.vectorizer = lambda x: self.vec.vectors[x]

        self.target = data[data_target]
        features = [front_padding(encoder(preprocessing(sequence), self.vec), max_seq_len) for sequence in data['comment_text'].tolist()]
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, i):
        assert len(self.features[i]) == self.max_seq_len
        return self.features[i], self.target[i]

###########################################################################################################################################################
###########################################################################################################################################################
train_df, test_df = train_test_split(data_toxic)

train_dataset = Data(train_df, data_target = 'toxic')
test_dataset = Data(test_df, data_target = 'toxic')


def traincollate(batch, vectorizer= train_dataset.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor([item[1] for item in batch]) 
    return inputs, target


def testcollate(batch, vectorizer = test_dataset.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor([item[1] for item in batch]) 
    return inputs, target



batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn= traincollate, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn= testcollate)


#####################################################################################################################################################
#####################################################################################################################################################



# print()
# print(dataset[0])
# print(dataset[10])

# from torch import nn
# import torch.optim as optim 

# emb_dim = 300

# model = ToxicClassifier(max_seq_len=32, emb_dim=emb_dim, hidden=32)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.003)



# epochs = 100

# for e in range(epochs):
#      running_loss = 0
#      for features, target in iter(train_loader):
#           features.resize_(features.size()[0], 32 * emb_dim)
#           optimizer.zero_grad()
#           output = model.forward(features)  
#           loss = criterion(output, target)
#           loss.backward()                  
#           optimizer.step()                
#           running_loss += loss.item()
     
#      print(f'Epoch:   {e+1}/{epochs}     Train Loss:  : {running_loss/len(train_loader):15}')