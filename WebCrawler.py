import pickle
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import glob


nlp = spacy.load('en_core_web_sm')

def fix_url(url):
	url = url.replace("/", "_")
	if 'https:' in url:
		url = url[len('https:'):]
	url = re.sub(r'\W+', '_', url)
	url = url.lstrip('_')
	return url

# 1, 2
cf = 0

def extract_text_from_url(url):
	try:
		response = requests.get(url)
	except:
		response = requests.get("https://en.wikipedia.org/wiki/Houston_Astros")
	soup = BeautifulSoup(response.text, 'html.parser')
	text = ' '.join([p.get_text() for p in soup.find_all('p')])
	return text

def write_text_to_file(text, url, cleaned, original):
	if cleaned:
		try:
			with open(f'cleaned_{fix_url(url)}.txt', 'w') as file:
				file.write(str(text.encode('utf-8', errors='replace')))
		except:
			print("wttf error")
	else:
		if original:
			try:
				with open(f'original_{fix_url(url)}.txt', 'w') as file:
					file.write(str(text.encode('utf-8', errors='replace')))
			except:
				print("wttf error")
		else:
			try:
				with open(f'{fix_url(url)}.txt', 'w') as file:
					file.write(str(text.encode('utf-8', errors='replace')))
			except:
				print("wttf error")
	
# 3

def clean_text(text):
	doc = nlp(text)
	cleaned_tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
	return ' '.join(cleaned_tokens)

def clean_text_files(urls):
	for url in urls:
		try:
			with open(f'{fix_url(url)}.txt', 'r') as file:
				text = file.read()
				cleaned_text = clean_text(text)
				write_text_to_file(cleaned_text, url, True, False)
		except:
			cf += 1

# 4

def extract_important_terms(texts):
	tfidf = TfidfVectorizer()
	X = tfidf.fit_transform(texts)
	feature_array = np.array(tfidf.get_feature_names_out())
	tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
	important_terms = tfidf.get_feature_names_out()
	return feature_array[tfidf_sorting][:40]

def get_important_terms_from_files(urls):
	texts = []
	my_texts = []
	for url in urls:
		with open(f'cleaned_{fix_url(url)}.txt', 'r') as file:
			text = file.read()
			texts.append(text)
		try:
			with open(f'original_{fix_url(url)}.txt', 'r') as file:
				text = file.read()
				my_texts.append(text)
		except:
			continue
	important_terms = extract_important_terms(my_texts)
	return important_terms

def find_context(combined_text, term):
	sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', combined_text)
	out = []
	for sentence in sentences:
		if term.lower() in sentence.lower():
			out.append(sentence)
	return out


def main():
	original_urls = ['https://en.wikipedia.org/wiki/Houston_Astros', 'https://www.mlb.com/astros', 'https://www.baseball-reference.com/teams/HOU/index.shtml']
	
	for url in original_urls:
		text = extract_text_from_url(url)
		write_text_to_file(text, url, False, True)
		write_text_to_file(text, url, False, False)

	# crawling for more URLs
	crawled_urls = []
	while len(crawled_urls) < 25:
		for original_url in original_urls:
			response = requests.get(original_url)
			if response.status_code == 200:
				soup = BeautifulSoup(response.content, 'html.parser')
				links = soup.find_all('a')
				explored_urls = [link.get('href') for link in links if link.get('href')]
				crawled_urls.extend(list(set(filter(lambda x: re.match(r'https?://', x), explored_urls))))
	crawled_urls = crawled_urls[:25]
	urls = original_urls + crawled_urls

	for url in urls:
		text = extract_text_from_url(url)
		if url in original_urls:
			write_text_to_file(text, url, False, True)
		else:
			write_text_to_file(text, url, False, False)	
	clean_text_files(urls)
	
	# top 40 terms
	important_terms = get_important_terms_from_files(urls)
	print("My top 40 terms are:", ', '.join(important_terms))

	# manually selecting top 15 terms
	top15_terms = ['sports', 'data', 'statistics', 'stats', 'calculations', 'historical', 'logos', 'david', 'property', 'llc', 'game', 'baseball', 'organization', 'houston', 'starters']
	print("My (manually selected) top 15 terms are:", ', '.join(top15_terms))

	# knowledge base construction
	kb = {}
	combined_text, OG_text = "", ""
	for file in glob.glob('[!cleaned]*.txt'):
		with open(file, 'r') as f:
			combined_text += f.read() + ' '
	for file in glob.glob('original*.txt'):
		with open(file, 'r') as f:
			OG_text += f.read() + ' '
	# definitions
	for term in important_terms:
		if len(wordnet.synsets(term)):
			kb[(term, "definitions")] = wordnet.synsets(term)[0].definition()
	# contexts
	for term in important_terms:
		kb[(term, "contexts")] = find_context(combined_text, term)
	# combined text
	kb[("context")] = combined_text
	kb[("original")] = OG_text
	# pickling
	with open('kb.pickle', 'wb') as file:
		pickle.dump(kb, file, protocol=pickle.HIGHEST_PROTOCOL)
	
	# For making the report:
		
	# for k, v in kb.items():
	# 	if k != ("context") and k != ("original"):
	# 		print(k, v)	

if __name__ == "__main__":
    main()
