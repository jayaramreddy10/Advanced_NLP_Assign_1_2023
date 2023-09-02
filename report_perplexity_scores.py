import numpy as np
import math
import spacy
import torch
from Problem_1_NNLM import *
from Problem_2_LSTM import LSTM_LM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_perplexity_scores(model, dataset, word_emb_mode, vocabulary):
    vocab_size = len(vocabulary)
    for sent_no, sentence in enumerate(dataset):
        # Tokenize and convert words to indices (replace with your preprocessing)
        words = clean_text(dataset)  #split sentence into words
        words_processed = process_data(val_data, vocabulary) 
        # = [word_emb_model[word] for word in words]  # Convert words to indices
        
        # Convert to PyTorch tensor
        inputs = torch.Tensor(input_indices).unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        outputs, _ = model(inputs)
        
        # Define target (e.g., if you have a target sentence)
        target = torch.LongTensor(target_indices).unsqueeze(0)  # Add batch dimension
        
        # Calculate loss
        loss = nn.CrossEntropyLoss(outputs.view(-1, vocab_size), target.view(-1))

        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Print or store perplexity for the current sentence
        print(f"Perplexity for sentence '{sentence}': {perplexity.item()}")

# Define your sentences and corresponding perplexity scores
# split text into sentences
nlp_sentencizer = spacy.blank("en")
nlp_sentencizer.add_pipe("sentencizer")
nlp_sentencizer.max_length = 6400000
corpus = open('Auguste_Maquet.txt', 'r')
ds = corpus.read()
ds.replace("\n", "")
tokens = nlp_sentencizer(ds)

sentences = []
for sentence in tokens.sents:
    sentences.append(str(sentence).replace('\n', " "))
print(len(sentences))

# split sentences to train and text
# random.shuffle(sentences)
train = sentences[:30000]
val = sentences[30000:40000]
test = sentences[40000:59710]
train = ''.join(str(train))   #combine all train sentences into single string
val = ''.join(str(val))   #combine all train sentences into single string
test = ''.join(str(test))     #combine all test sentences  into single string

# process data (split train and test data into words)
train_data = clean_text(train)  #split into words
val_data = clean_text(val)
test_data = clean_text(test)
vocabulary = build_vocab(train_data)  # give a label to each unique word 
vocabulary['unk'] = len(vocabulary)
word_emd_model = t_vocab.GloVe(name='6B', dim=300)  # You can change 'dim' to match the dimensionality of your GloVe model

is_NNLM = False
if(is_NNLM):
    # load model
    checkpoint_path = "/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_1/logs/NNLM/state_0.pt"
    # Create the model and optimizer objects
    #load model architecture 
    model = NNLM(input_dim = 300, hidden_dim_1 = 200, hidden_dim_2 = 25, len_vocab=len(vocabulary), window_size = 5)
    model = model.to(device)
else: 
    checkpoint_path = "/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_1/logs/LSTM/state_0.pt"
    # Create the model and optimizer objects
    #load model architecture 
        #load model architecture 
    vocab_size = len(vocabulary)  # Size of your vocabulary
    embedding_dim = 300  # Dimension of word embeddings
    hidden_dim = 512  # Dimension of hidden state in LSTM
    num_layers = 1  # Number of LSTM layers

    # Input shape: [seq_len : 5, batch_size, embedding_dim]
    # Output shape: [seq_len, batch_size, hidden_size]
    # Hidden shape: [num_layers : 2, batch_size, hidden_size]
    LM_model = LSTM_LM(embedding_dim, hidden_dim, num_layers, vocab_size)
    model = LM_model.to(device)

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)
# Access the desired variables from the checkpoint
model_state_dict = checkpoint['model']
# optimizer_state_dict = checkpoint['optimizer']
# epoch = checkpoint['epoch']


# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# Load the model and optimizer states from the checkpoint
model.load_state_dict(model_state_dict)

# compute PPL Scores
# train
train_PPL_scores = compute_perplexity_scores(model, train, word_emd_model, vocabulary)
# Calculate the average perplexity score
average_score = sum(train_PPL_scores) / len(train_PPL_scores)
# Define the file path where you want to save the text file
file_path = "perplexity_scores_train_NNLM.txt"
# Open the file in write mode and write the information
with open(file_path, "w") as file:
    for sentence, score in zip(train, train_PPL_scores):
        file.write(f'{sentence}\t{score:.2f}\n')
    file.write(f'Average Score: {average_score:.2f}')
# Print the average score
print(f'Average train PPL Score: {average_score:.2f}')

# val
val_PPL_scores = compute_perplexity_scores(model, val)
# Calculate the average perplexity score
average_score = sum(val_PPL_scores) / len(val_PPL_scores)
# Define the file path where you want to save the text file
file_path = "perplexity_scores_val_NNLM.txt"
# Open the file in write mode and write the information
with open(file_path, "w") as file:
    for sentence, score in zip(val, val_PPL_scores):
        file.write(f'{sentence}\t{score:.2f}\n')
    file.write(f'Average Score: {average_score:.2f}')
# Print the average score
print(f'Average val PPL Score: {average_score:.2f}')

# train
test_PPL_scores = compute_perplexity_scores(model, test)
# Calculate the average perplexity score
average_score = sum(test_PPL_scores) / len(test_PPL_scores)
# Define the file path where you want to save the text file
file_path = "perplexity_scores_test_NNLM.txt"
# Open the file in write mode and write the information
with open(file_path, "w") as file:
    for sentence, score in zip(test, test_PPL_scores):
        file.write(f'{sentence}\t{score:.2f}\n')
    file.write(f'Average Score: {average_score:.2f}')
# Print the average score
print(f'Average test PPL Score: {average_score:.2f}')