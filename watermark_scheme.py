import torch
import random
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import torch
import torch.nn.functional as F
from torch import nn
import hashlib
from scipy.stats import norm
import gensim
import pdb
from transformers import LlamaTokenizer, AutoModelForSequenceClassification, LlamaForCausalLM
from transformers import RobertaForSequenceClassification, RobertaTokenizer,BertTokenizer
import gensim.downloader as api
import Levenshtein
import string
import spacy
import paddle
from jieba import posseg

paddle.enable_static()
import re

class FormatWatermark:
    
    def __init__(self, p = float):
        self.withspace1 = "\u0020"
        self.withspace2 = "\u2004"
        self.probability = p


    def injection(self,text):
        
        output_text = ""
        for cur in text:
            if cur == self.withspace1:  
                p = random.random()
                if p <= self.probability:
                    cur = self.withspace2
            output_text = output_text + cur

        return str(output_text)
    
    def detection(self,text = str, p_value: float = 0.05) -> bool:
        count1 = text.count(self.withspace1) + text.count(self.withspace2)
        count2 = text.count(self.withspace2)
        if count2 == 0 or count1 == 0:
            return False
        ans = count2 / count1
        if (abs(self.probability - ans) < p_value):
            return True
        else:
            return False

class UniSpachWatermark:
    
    def __init__(self, p = float):
        self.whitespace = "\u0020"
        self.code = ["\u2000", "\u2001", "\u2004", "\u2006", "\u2007", "\u2008", "\u2009", "\u200A"]
        self.probability = p
    
    def injection(self, text = str) -> str:
        output_text = ""
        for cur in text:
            if cur == self.whitespace:
                p = random.random()
                if p <= self.probability:
                    num = random.randint(0,7)
                    cur = self.code[num]
            output_text = output_text + cur
        return output_text
    
    def detection(self, text = str, p_value: float = 0.05) -> bool:
        count = 0
        for cur in self.code:
            count += text.count(cur)
        count1 = count + text.count(self.whitespace)
        if count1 == 0 or count == 0:
            return False
        ans = count / count1
    
        if abs(self.probability - ans) < p_value:
            return True
        else:
            return False

class LexicalWatermark:
    
    def __init__(self,config, tau_word: float = 0.75, tau_sent: float = 0.8, lamda: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = config.model_name
        self.lamda = lamda
        self.model = LlamaForCausalLM.from_pretrained(self.model_name, device_map = 'auto')
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.relatedness_model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli').to(self.device)
        self.relatedness_tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        self.w2v_model = api.load("glove-wiki-gigaword-100")
        nltk.download('stopwords')
        self.tau_word = tau_word
        self.tau_sent = tau_sent
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm')
        self.en_tag_white_list = set(['MD', 'NN', 'NNS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS'])
    def is_subword(self,token: str): 
        return token.startswith('##')

    def binary_encoding_function(self,token):
        hash_value = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16)
        random_bit = hash_value % 2
        return random_bit
    def is_similar(self,x, y, threshold=0.5):
        distance = Levenshtein.distance(x, y)
        if distance / max(len(x), len(y)) < threshold:
            return True
        return False
    def sent_tokenize(self,ori_text):
        return nltk.sent_tokenize(ori_text)
    
    def global_word_sim(self,word,ori_word):
        try:
            global_score = self.w2v_model.similarity(word,ori_word)
        except KeyError:
            global_score = 0
        return global_score
    
    def pos_filter(self, tokens, masked_token_index, input_text):
        pos_tags = pos_tag(tokens)
        pos = pos_tags[masked_token_index][1]
        if pos not in self.en_tag_white_list:
            return False
        if self.is_subword(tokens[masked_token_index]) or self.is_subword(tokens[masked_token_index+1]) or (tokens[masked_token_index] in self.stop_words or tokens[masked_token_index] in string.punctuation):
            return False
        return True
    def context_word_sim(self,init_candidates, tokens, masked_token_index, input_text):
        original_input_tensor = self.tokenizer.encode(input_text,return_tensors='pt').to(self.device)
        batch_input_ids = [[self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens[2:masked_token_index] + [token] + tokens[masked_token_index+1:-1]+ ['[SEP]'])] for token in init_candidates]
        batch_input_tensors = torch.tensor(batch_input_ids).squeeze().to(self.device)
        batch_input_tensors = torch.cat((batch_input_tensors,original_input_tensor),dim=0)
        with torch.no_grad():
            outputs = self.model(batch_input_tensors, output_hidden_states = True)
            cos_sims = torch.zeros([len(init_candidates)]).to(self.device)
            embedding_layers = outputs.hidden_states
            layers = len(embedding_layers)
            N = 8
            i = masked_token_index
            cos_sim_sum = 0
            for num in range(layers-N,layers):
                temp_hidden_states = embedding_layers[num]
                ls_hidden_states = temp_hidden_states[0:len(init_candidates), int(i), :]
                source_hidden_state = temp_hidden_states[len(init_candidates), i, :]
                cos_sim_sum += F.cosine_similarity(source_hidden_state, ls_hidden_states, dim=1)
            cos_sim_avg = cos_sim_sum / N
            
            cos_sims += cos_sim_avg
        return cos_sims.tolist()
    def filter_special_candidate(self, top_n_tokens, tokens,masked_token_index):
        filtered_tokens = [tok for tok in top_n_tokens if tok not in self.stop_words and tok not in string.punctuation and pos_tag([tok])[0][1] in self.en_tag_white_list and not self.is_subword(tok)]

            # for token in filtered_tokens:
            #     doc = self.nlp(token)
            #     lemma = doc[0].lemma_ if doc[0].lemma_ != "-PRON-" else token
            #     lemmatized_tokens.append(lemma)
            
        base_word = tokens[masked_token_index] 
        base_word_lemma = self.nlp(base_word)[0].lemma_ 
        processed_tokens = [base_word]+[tok for tok in filtered_tokens if self.nlp(tok)[0].lemma_ != base_word_lemma]
        return processed_tokens
        
    def candidates_gen(self,tokens,masked_token_index,input_text,topk=64, dropout_prob=0.3):
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if not self.pos_filter(tokens,masked_token_index,input_text):
            return []
        # Create a tensor of input IDs
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor, output_hidden_states = True)
            embeddings = outputs.hidden_states[0]

        dropout = nn.Dropout2d(p=dropout_prob)
        # Get the predicted logits
        embeddings[:, masked_token_index, :] = dropout(embeddings[:, masked_token_index, :])
        with torch.no_grad():
            outputs = self.model(inputs_embeds=embeddings)
        predicted_logits = outputs[0][0][masked_token_index]

        # Set the number of top predictions to return
        n = topk
        # Get the top n predicted tokens and their probabilities
        probs = torch.nn.functional.softmax(predicted_logits, dim=-1)
        top_n_probs, top_n_indices = torch.topk(probs, n)
        top_n_tokens = self.tokenizer.convert_ids_to_tokens(top_n_indices.tolist())
        processed_tokens = self.filter_special_candidate(top_n_tokens,tokens,masked_token_index)
          
        return processed_tokens
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string
    
    def sentence_sim(self,init_candidates, tokens, masked_token_index, input_text):
        batch_sentences = [self.convert_tokens_to_string(tokens[1:masked_token_index] + [token] + tokens[masked_token_index+1:-1]) for token in init_candidates]
        roberta_inputs = [input_text + '</s></s>' + s for s in batch_sentences]
        
        encoded_dict = self.relatedness_tokenizer.batch_encode_plus(
                roberta_inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt')
        # Extract input_ids and attention_masks
        input_ids = encoded_dict['input_ids'].to(self.device)
        attention_masks = encoded_dict['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.relatedness_model(input_ids=input_ids, attention_mask=attention_masks)
            logits = outputs[0]
        probs = torch.softmax(logits, dim=1)
        relatedness_scores = probs[:, 2].tolist()
        
        return relatedness_scores
    def filter_candidates(self, init_candidates, tokens, masked_token_index, input_text):
        context_word_similarity_scores = self.context_word_sim(init_candidates, tokens, masked_token_index, input_text)
        sentence_similarity_scores = self.sentence_sim(init_candidates, tokens, masked_token_index, input_text)
        filtered_candidates = []
        for idx, candidate in enumerate(init_candidates):
            global_word_similarity_score = self.global_word_sim(tokens[masked_token_index], candidate)
            word_similarity_score = self.lamda*context_word_similarity_scores[idx]+(1-self.lamda)*global_word_similarity_score
            if word_similarity_score >= self.tau_word and sentence_similarity_scores[idx] >= self.tau_sent:
                filtered_candidates.append((candidate, word_similarity_score))#, sentence_similarity_scores[idx]))
        return filtered_candidates
    
    def injection(self,text):
        input_text = text
        # Tokenize the input text
        tokens = self.tokenizer.tokenize(input_text) 
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        masked_tokens=tokens.copy()
        start_index = 1
        end_index = len(tokens) - 1
        for masked_token_index in range(start_index+1, end_index-1):
            # pdb.set_trace()
            temp_tokens = tokens[masked_token_index - 1] + tokens[masked_token_index]
            binary_encoding = self.binary_encoding_function(temp_tokens)
            if binary_encoding == 1:
                continue
            init_candidates = self.candidates_gen(tokens,masked_token_index,input_text, 32, 0.3)
            if len(init_candidates) <=1:
                continue
            enhanced_candidates = self.filter_candidates(init_candidates,tokens,masked_token_index,input_text)
            hash_top_tokens = enhanced_candidates.copy()  
            for i, tok in enumerate(enhanced_candidates):
                temp_tokens = tokens[masked_token_index - 1] + tok[0]
                binary_encoding = self.binary_encoding_function(temp_tokens)
                if binary_encoding != 1 or (self.is_similar(tok[0], tokens[masked_token_index])) or (tokens[masked_token_index - 1] in tok or tokens[masked_token_index + 1] in tok):   
                    hash_top_tokens.remove(tok)                
            hash_top_tokens.sort(key=lambda x: x[1], reverse=True)    
            if len(hash_top_tokens) > 0:
                selected_token = hash_top_tokens[0][0]
            else:
                selected_token = tokens[masked_token_index]
            
            tokens[masked_token_index] = selected_token
        watermarked_text = " ".join(tokens).replace(" ##", "").strip()
        return watermarked_text
    def get_encodings_fast(self,text):
        sents = self.sent_tokenize(text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)
        encodings = []
        for i in range(0, num_sents, 2):
            if i+1 < num_sents:
                sent_pair = sents[i] + sents[i+1]
            else:
                sent_pair = sents[i]
            tokens = self.tokenizer.tokenize(sent_pair)
            
            for index in range(1,len(tokens)-1):
                if not self.pos_filter(tokens,index,text):
                    continue
                temp_tokens = tokens[index-1]+tokens[index]
                bit = self.binary_encoding_function(temp_tokens)
                encodings.append(bit)
        return encodings
    
    def embed(self, ori_text):
        sents = self.sent_tokenize(ori_text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)
        watermarked_text = ''
        for i in range(0, num_sents, 2):
            if i+1 < num_sents:
                sent_pair = sents[i] + sents[i+1]
            else:
                sent_pair = sents[i]
            if len(watermarked_text) == 0:
                watermarked_text = self.injection(sent_pair)
            else:
                watermarked_text = watermarked_text + self.injection(sent_pair)
        if len(self.get_encodings_fast(ori_text)) == 0:
            return ''
        return watermarked_text
    
    def get_encodings_precise(self, text):
        sents = self.sent_tokenize(text)
        sents = [s for s in sents if s.strip()]
        num_sents = len(sents)
        encodings = []
        for i in range(0, num_sents, 2):
            if i+1 < num_sents:
                sent_pair = sents[i] + sents[i+1]
            else:
                sent_pair = sents[i]

            tokens = self.tokenizer.tokenize(sent_pair) 
            
            tokens = ['[CLS]'] + tokens + ['[SEP]']

            masked_tokens=tokens.copy()

            start_index = 1
            end_index = len(tokens) - 1

            for masked_token_index in range(start_index+1, end_index-1):
                init_candidates = self.candidates_gen(tokens,masked_token_index,sent_pair, 8, 0)        
                if len(init_candidates) <=1:
                    continue
                enhanced_candidates = self.filter_candidates(init_candidates,tokens,masked_token_index,sent_pair)      
                if len(enhanced_candidates) > 1:
                    temp_tokens = tokens[masked_token_index-1]+tokens[masked_token_index]
                    bit = self.binary_encoding_function(temp_tokens)
                    encodings.append(bit)
        return encodings
    
    def detection(self,text,p_value=0.05):
        p = 0.5
        encodings = self.get_encodings_precise(text)
        n = len(encodings)
        ones = sum(encodings)
        if n == 0:
            z = 0 
        else:
            z = (ones - p * n) / (n * p * (1 - p)) ** 0.5
        threshold = norm.ppf(1 - p_value, loc=0, scale=1)
        p_value = norm.sf(z)
        is_watermark = z >= threshold
        return is_watermark
    


