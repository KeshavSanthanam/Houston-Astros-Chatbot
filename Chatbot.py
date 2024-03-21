import sys
import random
import nltk
nltk.download('sentiwordnet')
nltk.download('wordnet')
from nltk.metrics.distance import jaccard_distance 
from nltk.util import ngrams
nltk.download('words') 
from nltk.corpus import words 
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import sentiwordnet
import pickle
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

wnl = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

# retrieving knowledge base
kb = {}
with open('kb.pickle', 'rb') as file:
    kb = pickle.load(file)

prev_users = []

# text for later tokenization
knowledge_text = ""
context_text = ""
my_text = ""
for key, val in kb.items():
	if key == ("context"):
		context_text = val
	elif key == ("original"):
		my_text = val
	else:
		knowledge_text += str(val) + ' '	

greetings = ['Hello! My name is ChatBot. How can I help you today?', 'Hey, I\'m ChatBot! What\'s up?', 'Good day! ChatBot is here to assist you. What do you need help with today?']

# prompting user
def prompt_user(user=prev_users[-1] if prev_users else None):
	def autocorrect(s):
		output_string = ""
		for word in s.split(): 
			try:
				temp = [(jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))),w) for w in words.words() if w[0]==word[0]]
				output_string += str(sorted(temp, key = lambda val:val[0])[0][1] + " ")
			except ZeroDivisionError:
				output_string += word
		return output_string

	def processed(s):
		output_string = ""
		list_s = s.split()
		s = ' '.join([wnl.lemmatize(word) for word in list_s])
		tokens = nlp(s)
		for token in tokens:
			if not token.is_stop and token.is_alpha:
				output_string += token.text.lower() + ' '
		return output_string.strip()
	
	print(random.choice(greetings))
	print("**Thanks for using ChatBot.**")
	print("**Program terminates when input is empty/contains \"Goodbye ChatBot\" (case-insensitive)**")
	print("**If you'd like, SAY YOUR NAME in the following format: My name is ____ ...**")
	prompt = ""
	name = ""
	prompts_list = []
	prompt = input()
	while len(prompt) == 0:
		print("Prompt is empty. Try again!")
		prompt = input()
	prompt = prompt.lower()
	if "goodbye chatbot" not in prompt:
		if "my name is " in prompt:
			name = prompt[prompt.find("my name is ") + len("my name is "):].strip().lower().split()[0]
			print(f"Hello {name}! I am storing user data tied to your identity :)")
			for old_user in prev_users:
				if old_user.name == "name":
					user = old_user
			if not user:
				user = User(name, len(prompt))
			prompt = prompt[prompt.find("my name is ") + len("my name is ") + len(name):].strip()
		prompts_list.append((prompt, processed(prompt)))
	else:
		print("Bye! See you soon.")
		exit(0)
	evaluate_prompt(prompts_list[-1], user)
	return prompts_list
	
# prompt evaluation
def evaluate_prompt(p_and_p, user):
	def partial_search_dict(d, s):
		for key in d.keys():
			for elem in key:
				if s in elem:
					return d[key]
		return None

	def text_search(combined_text, term):
		sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', combined_text)
		for sentence in sentences:
			if term.lower() in sentence.lower():
				return sentence
		return ""

	def filtered(text):
		text = str(re.sub(r'[^\x00-\x7f]', '', text))
		puncts = ['.', '!', '?']
		for i, char in enumerate(text):
			if char in puncts:
				text = text[:i+1]
		esc_seqs = ''.join([chr(char) for char in range(1, 32)])
		text = text.translate(str.maketrans('', '', esc_seqs)) 
		text = text.replace('\\t', '').replace('\\n', '').replace('\\r', '').replace('\\f', '').replace('\\b', '')
		if text[:2] == 'b\'':
			text = text[2:]
		return text.encode().decode('unicode_escape')
	
	def most_imp_noun(prompt):
		tokens = word_tokenize(prompt)
		tags = pos_tag(tokens)
		nouns = [word for word, pos in tags if pos.startswith('NN')]
		if not nouns:
			return (prompt, prompt)
		nouns_combined = ' '.join(nouns)
		tfidf = TfidfVectorizer()
		baseline = tfidf.fit_transform([nouns_combined])
		nouns_combined = ', '.join(nouns)
		return (tfidf.get_feature_names_out()[baseline.argmax()], nouns_combined)

	prompt, proc_prompt = p_and_p
	unsupported_messages = ["draw me a", "tell me a story", "change topics"]
	for msg in unsupported_messages:
		if msg in prompt:
			print("ChatBot can only support certain kinds of requests related to the predefined topic. Try again.")
			prompt_user(user)
	
	# now we generate a response based on their input and our knowledge base
	response = ""
	explainers = ["Here's something I know about ", "Let me tell you about ", "Here's a topic I know is relevant: "]
	contexts = ["Something worth mentioning is that ", "A relevant piece of context: ", "By the way, "]
	prompt, nouns_combined = most_imp_noun(prompt)
	while prompt and "goodbye chatbot" not in prompt:
		user_tokens = (word_tokenize(prompt), word_tokenize(proc_prompt))
		KT_tokens, context_tokens = word_tokenize(knowledge_text), word_tokenize(context_text)
		overlap = set(KT_tokens).union(set(context_tokens))
		for token in user_tokens[0]:
			kb_val = text_search(my_text.lower(), token)			
			if kb_val == None:
				kb_val = str(partial_search_dict(kb, token))
			if not kb_val:
				continue
			kb_val = kb_val.encode('ascii', 'ignore').decode("utf-8")
			if len(prompt.split()) <= 3:
				about_what = nouns_combined
			else:
				about_what = str(token.strip())
			response = f"{random.choice(explainers)}{about_what}. {filtered(kb_val)}"
		if not response:
			print("Sorry! I don't have information on this in my knowledge base.")
		print(response)
		if user and user.user_likes:
			print("I can recommend you think about ", random.choice(user.user_likes))
		preserved_prompt = prompt
		prompt = ""

	# finding likes and dislikes 
	def user_sentiment(prompt):
		def find_pos_wn(tag):
			if tag.startswith('J'):
				return wordnet.ADJ
			elif tag.startswith('V'):
				return wordnet.VERB
			elif tag.startswith('N'):
				return wordnet.NOUN
			elif tag.startswith('R'):
				return wordnet.ADV
			return ""

		tokens = word_tokenize(prompt)
		pos_tags = pos_tag(tokens)
		sentiment, num_tokens = 0, 0
		for word, pos in pos_tags:
			my_tag = find_pos_wn(pos)
			if my_tag not in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV, wordnet.VERB): continue
			lemma = wnl.lemmatize(word, pos=my_tag)
			if not lemma: continue
			synsets = wordnet.synsets(lemma, pos=my_tag)
			if not synsets: continue
			for synset in synsets:
				synset_name = synset.name()
				swn_synset = sentiwordnet.senti_synset(synset_name)
				if swn_synset:
					sentiment += (swn_synset.pos_score() - swn_synset.neg_score())
					num_tokens += 1
			if num_tokens > 0:
				sentiment /= num_tokens
		return sentiment
	
	# update user information
	if user:
		if user_sentiment(preserved_prompt) > 0:
			user.update_info(user.name, len(preserved_prompt), [nouns_combined], [])
		else:
			user.update_info(user.name, len(preserved_prompt), [], [nouns_combined])
	prompt_user(user)

# class for maintaining user data
class User:

	def __init__(self, name, words_discussed, user_likes=[], user_dislikes=[]):
		self.name = name
		self.words_discussed = words_discussed
		self.user_likes = user_likes
		self.user_dislikes = user_dislikes

		with open(f'{name.replace("/", "_")}.txt', 'w') as file:
			file.write("User name: " + self.name + "\n")
			file.write("Words discussed: " + str(self.words_discussed) + "\n")
			file.write("User likes: " + ', '.join(self.user_likes) + "\n")
			file.write("User dislikes: " + ', '.join(self.user_dislikes) + "\n")

	def update_info(self, name, words_discussed, user_likes, user_dislikes):
		self.name = name
		self.words_discussed += words_discussed
		self.user_likes += user_likes
		self.user_dislikes += user_dislikes
		with open(f'{name.replace("/", "_")}.txt', 'w') as file:
			file.write("User name: " + self.name + "\n")
			file.write("Words discussed: " + str(self.words_discussed) + "\n")
			file.write("User likes: " + ', '.join(self.user_likes) + "\n")
			file.write("User dislikes: " + ', '.join(self.user_dislikes) + "\n")


prompts_list = prompt_user()

