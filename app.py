import streamlit as st
import asyncio
import requests_cache
import lxml
import re
from bs4 import BeautifulSoup
from bs4 import NavigableString
import json
import pandas as pd
import spacy
from spacy import displacy
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import numpy as np
from fake_useragent import UserAgent
from collections import Counter
import itertools as it
import networkx as nx
from urllib.parse import urlparse
from nltk.corpus import wordnet
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import tensorflow as tf
import wikidata
from wikidataintegrator import wdi_core
from wikidataintegrator import wdi_login
import scattertext as st
from sklearn.feature_extraction import _stop_words
import io
from scipy.stats import rankdata, hmean, norm
import os, pkgutil, urllib
from urllib.request import urlopen
from IPython.display import IFrame, display, HTML
from urllib.parse import urlparse, urlsplit
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
import logging
import time


ua = UserAgent()
# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


# Load pre-trained models and initialize variables here.
try:
    nlp = spacy.load("en_core_web_md")
except:
  !python -m spacy download en_core_web_md
  nlp = spacy.load("en_core_web_md")

# نموذج تحليل المشاعر
sentiment_model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
sentiment_model_revision = "714eb0f"
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model_name, revision=sentiment_model_revision)

# تحميل نموذج Sentence-BERT
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

#Load text generation model
text_generator = pipeline("text-generation", model="gpt2")


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('punkt_tab')

# Classes for better code organization.
class ContentExtractor:
    def __init__(self, ua):
        self.ua = ua
        self.session = requests_cache.CachedSession('cached_requests')
        self.min_content_length = 50
        self.min_heading_length = 5
        self.unwanted_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d+(\.\d+)?(/5|%)',
            r'[★☆]+',
            r'(\b\d+\b\s*){3,}'
        ]

    async def fetch_url(self, url):
      try:
          async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={'User-Agent': self.ua.random}) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logging.error(f"Error: Status code {response.status} for {url}")
                    return None
      except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None


    async def fetch_all_urls(self, urls):
        tasks = [self.fetch_url(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def extract_content(self, links):
        titles = []
        headings = []
        content = []
        lists = []
        tables = []
        meta_descriptions = []
        internal_links = []
        external_links = []
        schema_org_data = []
        open_graph_data = []
        json_ld_data = []

        html_responses = await self.fetch_all_urls(links)

        for url, response in zip(links, html_responses):
            if response:
                try:
                    print("جاري معالجة الرابط:", url)
                    soup = BeautifulSoup(response, 'lxml')

                    temp_titles = soup.find_all('title')
                    temp_content = soup.find_all('p')
                    temp_headings = soup.find_all(re.compile('^h[1-4]$'))
                    temp_lists = soup.find_all(['ul', 'ol'])
                    temp_tables = soup.find_all('table')
                    temp_meta_desc = soup.find_all('meta', attrs={'name':'description'})
                    temp_links = soup.find_all('a', href=True)

                    current_titles = []
                    for title in temp_titles:
                        if title.text and title.text.strip() and len(title.text.split()) >= self.min_heading_length and not any(re.search(pattern, title.text) for pattern in self.unwanted_patterns):
                           current_titles.append(title.text)

                    current_content = []
                    for paragraph in temp_content:
                        if paragraph.text and paragraph.text.strip() and len(paragraph.text.split()) >= self.min_content_length and not any(re.search(pattern, paragraph.text) for pattern in self.unwanted_patterns):
                          current_content.append(paragraph.text)

                    current_headings = []
                    for heading in temp_headings:
                        if heading.text and heading.text.strip() and len(heading.text.split()) >= self.min_heading_length and not any(re.search(pattern, heading.text) for pattern in self.unwanted_patterns):
                          current_headings.append(heading.text)

                    current_lists = [lst.text.strip() for lst in temp_lists]
                    current_tables = [table.text.strip() for table in temp_tables]
                    current_meta_desc = [meta.get('content').strip() for meta in temp_meta_desc if meta.get('content') and meta.get('content').strip()]
                    current_internal_links = [link.get('href') for link in temp_links if link.get('href') and url in link.get('href')]
                    current_external_links = [link.get('href') for link in temp_links if link.get('href') and (link.get('href').startswith("http://") or link.get('href').startswith("https://")) and url not in link.get('href') ]

                    titles.extend(current_titles)
                    content.extend(current_content)
                    headings.extend(current_headings)
                    lists.extend(current_lists)
                    tables.extend(current_tables)
                    meta_descriptions.extend(current_meta_desc)
                    internal_links.extend(current_internal_links)
                    external_links.extend(current_external_links)

                    # Extracting schema.org data
                    schema_data = soup.find_all('script', type='application/ld+json')
                    current_schema_org_data = []
                    for script in schema_data:
                        try:
                            current_schema_org_data.append(json.loads(script.string))
                        except Exception as e:
                          logging.error(f"Error extracting schema.org data {e} from {url}")
                          continue
                    schema_org_data.extend(current_schema_org_data)

                     # Extracting Open Graph data
                    og_tags = soup.find_all('meta', attrs={'property': re.compile('^og:')})
                    current_open_graph_data = {tag.get('property').replace("og:",""): tag.get('content') for tag in og_tags if tag.get('content')}
                    open_graph_data.append(current_open_graph_data)

                     # Extracting JSON-LD data
                    json_ld_scripts = soup.find_all('script', type='application/ld+json')
                    current_json_ld_data = []
                    for script in json_ld_scripts:
                        try:
                           current_json_ld_data.append(json.loads(script.string))
                        except Exception as e:
                            logging.error(f"Error extracting JSON-LD data {e} from {url}")
                            continue
                    json_ld_data.extend(current_json_ld_data)

                except Exception as e:
                    logging.error(f"حدث خطأ أثناء معالجة الرابط: {url}, {e}")
                    continue

        max_len = max(len(titles), len(content), len(headings), len(lists), len(tables), len(meta_descriptions), len(internal_links), len(external_links), len(schema_org_data), len(open_graph_data), len(json_ld_data))

        titles.extend([""]*(max_len - len(titles)))
        content.extend([""]*(max_len - len(content)))
        headings.extend([""]*(max_len - len(headings)))
        lists.extend([""]*(max_len - len(lists)))
        tables.extend([""]*(max_len - len(tables)))
        meta_descriptions.extend([""]*(max_len - len(meta_descriptions)))
        internal_links.extend([""]*(max_len - len(internal_links)))
        external_links.extend([""]*(max_len - len(external_links)))
        schema_org_data.extend([""]*(max_len - len(schema_org_data)))
        open_graph_data.extend([""]*(max_len - len(open_graph_data)))
        json_ld_data.extend([""]*(max_len - len(json_ld_data)))


        return titles, content, headings, lists, tables, meta_descriptions, internal_links, external_links, schema_org_data, open_graph_data, json_ld_data

class TextAnalyzer:
  def __init__(self):
    self.stop = stopwords.words('english')

  def get_sentence_embedding(self, text):
      inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
      with torch.no_grad():
          outputs = model(**inputs)
      embeddings = outputs.last_hidden_state.mean(dim=1)
      return embeddings.numpy()

  def lemmatize_texts(self, texts):
      docs = list(nlp.pipe(texts))
      lemmatized_output = [[token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']] for doc in docs]
      return lemmatized_output

  def get_pos_tags_batch(self, texts):
      docs = list(nlp.pipe(texts))
      pos_tags = [[(token.text, token.pos_) for token in doc] for doc in docs]
      return pos_tags

  def get_entities_batch(self, texts):
      docs = list(nlp.pipe(texts))
      entities = [[(ent.text, ent.label_) for ent in doc.ents] for doc in docs]
      return entities

  def filter_high_frequency_words(self, lemmatized_content, max_freq=0.8):
      word_counts = Counter(lemmatized_content)
      total_words = len(lemmatized_content)
      filtered_words = [word for word, count in word_counts.items() if count / total_words <= max_freq]
      return filtered_words

  def filter_unrelated_entities(self, entities, keywords, min_similarity=0.4):
      filtered_entities = []
      for ent, label in entities:
        for keyword in keywords:
          if keyword in ent.lower() or (nlp(ent).similarity(nlp(keyword)) > min_similarity):
            filtered_entities.append((ent, label))
            break
      return filtered_entities

  def analyze_sentiment(self, text):
      max_length = 512
      if len(text) > max_length:
          chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
          sentiment_results = []
          for chunk in chunks:
              results = sentiment_pipeline(chunk)
              if results and results[0]:
                sentiment_results.append((results[0]['label'], results[0]['score']))
          if sentiment_results:
            sentiment_labels = [label for label,score in sentiment_results]
            sentiment_label_counts = Counter(sentiment_labels)
            most_common_label = sentiment_label_counts.most_common(1)[0][0]
            sentiment_scores = [score for label,score in sentiment_results]
            average_score = sum(sentiment_scores)/len(sentiment_scores)
            return most_common_label, average_score
          else:
            return None, None
      else:
          results = sentiment_pipeline(text)
          if results and results[0]:
              return results[0]['label'], results[0]['score']
          else:
            return None, None

  def handle_missing_sentiment(self, sentiment, score):
    if not sentiment:
      return "Neutral", 0.5
    return sentiment, score

  def average_sentence_length(self, text):
    sentences = nltk.sent_tokenize(text)
    total_words = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)
    if len(sentences) > 0 :
        return total_words/ len(sentences)
    else:
        return 0

  def classify_entities(self, entities, keywords, root_threshold=2, rare_threshold=2):
      root_entities = []
      rare_entities = []
      unique_entities = []
      entity_counts = Counter(entities)
      login_instance = wdi_login.WDLogin(user="Alraadalmithali",pwd="q$Ae*wvsGE2!$jK") # please create a username and password at wikidata.org then provide credentials here.
      for ent, count in entity_counts.items():
        if any(keyword in ent for keyword in keywords):
          try:
            search_results = wdi_core.WDItemEngine.search_wikidata(ent)
            if search_results:
                entity_id = search_results[0]
                item = wdi_core.WDItemEngine(wd_item_id=entity_id)
                entity_type = item.wd_item_id
                if entity_type:
                  if count > root_threshold:
                      root_entities.append(ent)
                  elif count == rare_threshold:
                      rare_entities.append(ent)
                  elif count == 1:
                      unique_entities.append(ent)
                else:
                    if count > root_threshold:
                      root_entities.append(ent)
                    elif count == rare_threshold:
                      rare_entities.append(ent)
                    elif count == 1:
                      unique_entities.append(ent)

          except Exception as e:
              logging.error(f"Error classifying entities: {e}")
              if count > root_threshold:
                  root_entities.append(ent)
              elif count == rare_threshold:
                  rare_entities.append(ent)
              elif count == 1:
                  unique_entities.append(ent)

      return root_entities, rare_entities, unique_entities

  def extract_ngrams(self, text, n=3):
    words = [word for sent in text for word in sent]
    n_grams = nltk.ngrams(words, n)
    return list(n_grams)

  def count_ngrams(self, df, n=3, max_freq=0.8, min_freq=0.1):
      ngram_counts = Counter()
      for text in df:
        ngram_counts.update(self.extract_ngrams(text, n))
      total_ngrams = len(ngram_counts)
      filtered_ngrams = [(ngram, count) for ngram, count in ngram_counts.items() if (count/total_ngrams <= max_freq and count / total_ngrams >= min_freq)]
      return filtered_ngrams

  def filter_ngrams(self, ngrams):
      filtered_ngrams = [(ngram, count) for ngram, count in ngrams if all(pos not in ['CCONJ', 'ADP', 'PART', 'PRON'] for token,pos in pos_tag(ngram))]
      return filtered_ngrams

  def create_word_graph(self, df):
      noun_phrases = []
      for item in df:
        if item:
          valid_tokens = []
          for token_pos in item:
              if isinstance(token_pos, tuple) and len(token_pos) == 2:
                token, pos = token_pos
                if re.match(r'NN*|JJ*', pos):
                   valid_tokens.append(token)
          noun_phrases.append(valid_tokens)
        else:
          noun_phrases.append([])

      edges = [edge for phrase in noun_phrases for edge in it.combinations(phrase, 2) if phrase]

      G = nx.Graph(edges)
      index = nx.betweenness_centrality(G)
      for component in list(nx.connected_components(G)):
          if len(component)<5:
              for node in component:
                  G.remove_node(node)

      sorted_index = sorted(index.items(), key=lambda x:x[1], reverse=True)
      return sorted_index

  def create_semantic_network(self, pos_tags):
        semantic_edges = []
        for i in range(len(pos_tags) - 1):
            token1, pos1 = pos_tags[i]
            token2, pos2 = pos_tags[i + 1]
            if re.match(r'NN*|JJ*|VB*', pos1) and re.match(r'NN*|JJ*|VB*', pos2):
                semantic_edges.append((token1, token2))
        G = nx.DiGraph(semantic_edges)
        return G

  def identify_semantic_words(self, pos_tags, keywords):
        macro_keywords = [word for word, tag in pos_tags if tag in ['NOUN'] and word in keywords]
        micro_keywords = [word for word, tag in pos_tags if tag in ['ADJ', 'ADV', 'VERB'] and word not in keywords]
        return macro_keywords, micro_keywords


  def analyze_relationships(self, text):
    doc = nlp(text)
    relationships = []
    for ent in doc.ents:
      for other_ent in doc.ents:
        if ent != other_ent:
           if ent.start < other_ent.start:
              relationships.append((ent.text, other_ent.text))
    return relationships

  def summarize_text(self, text):
        max_length = 512
        if len(text) > max_length:
          chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
          summaries = []
          for chunk in chunks:
            try:
               summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
               summaries.append(summary)
            except Exception as e:
              logging.error(f"Error summarizing text: {e}")
              continue
          return " ".join(summaries) if summaries else None
        else:
          try:
               summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
               return summary
          except Exception as e:
            logging.error(f"Error summarizing text: {e}")
            return None

  def get_topics(self, texts, num_topics=5):
        tokens = [word_tokenize(text) for text in texts]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
        return lda_model.print_topics(num_words=5)

class ReportGenerator:

  def generate_headings(self, row, keywords):
        h1_template = "The ultimate guide to {}"
        h2_template = "Key strategies for {} issues like {}"
        micro_context_template = "Advanced methods for {}"

        # Analyze POS for keywords
        keyword_pos_tags = pos_tag(keywords)
        noun_keywords = [word for word, pos in keyword_pos_tags if pos.startswith('NN')]
        other_keywords = [word for word, pos in keyword_pos_tags if not pos.startswith('NN')]

        # Combine most frequent ngrams
        ngram_phrases = [" ".join(ngram) for ngram, count in row['Content_Ngrams_3'][:3]] if row['Content_Ngrams_3'] else []

        main_entity= ", ".join(row['Root_Entities'][:1]) if row['Root_Entities'] else ", ".join(noun_keywords[:1] if noun_keywords else keywords[:1] )

        h1 = h1_template.format(main_entity)
        if row['Rare_Entities']:
          h2s = [h2_template.format(" ".join(other_keywords) if other_keywords else "managing",  ", ".join(row['Rare_Entities'][:3]))]
        elif noun_keywords:
          h2s = [h2_template.format(" ".join(other_keywords) if other_keywords else "managing",  ", ".join(noun_keywords[:3]))]
        elif ngram_phrases:
           h2s = [h2_template.format(" ".join(other_keywords) if other_keywords else "managing",  ", ".join(ngram_phrases[:3]))]
        else:
          h2s = ["Key strategies for managing common issues."]

        micro_context_heading = micro_context_template.format(", ".join(row['Unique_Entities'][:3] + ngram_phrases)) if row['Unique_Entities'] or ngram_phrases else "Advanced methods for pest control."
        return h1, h2s, micro_context_heading

  def generate_methodology(self, row, keywords, country_code):
        methodology_template = """To create a comprehensive guide about {keywords} in {location}, you should focus on root issues like {root_entities}, provide clear actionable tips about {rare_entities} and finally offer advanced methods about {unique_entities}.
        You should consider the following:
            - The tone should be formal and authoritative.
            - The style should be descriptive and instructional.
            - Use keywords in the headings, subheadings and body paragraphs, and should be easy to understand for human readers.
            - Include information like relevant entities, dates, units, key phrases and lexical relations.
            - Always provide clear instructions about the format (paragraph or list).
            - For list items, give a list definition first.
            - Include the most relevant internal links in the micro context, linking to the best resources that covers the topic.
            - Consider using the following macro semantic words: {macro_keywords}.
            - Consider using the following micro semantic words: {micro_keywords}.
            - Good assertions are factual and based on evidence.
            - The average sentence length is: {sentence_length} words, it’s better if it remains below 20 words.
            - Use the following types of questions to formulate headings: definitional, grouping, comparative, boolean, representative, represented, and implicit questions.
        """

        methodology = methodology_template.format(
            keywords = ", ".join(keywords),
            location = country_code,
            root_entities =  ", ".join(row['Root_Entities'][:3]) if row['Root_Entities'] else "the main topic",
            rare_entities = ", ".join(row['Rare_Entities'][:3]) if row['Rare_Entities'] else "common issues",
            unique_entities = ", ".join(row['Unique_Entities'][:3] + [" ".join(ngram) for ngram, count in row['Content_Ngrams_3'][:2]] if row['Content_Ngrams_3'] else []) if row['Unique_Entities'] or row['Content_Ngrams_3'] else  "related issues",
            macro_keywords = ", ".join(row['Macro_Keywords'][:5]),
            micro_keywords = ", ".join(row['Micro_Keywords'][:10]),
             sentence_length = row['Average_Sentence_Length']
        ) if row['Root_Entities'] and row['Rare_Entities'] and row['Unique_Entities'] or row['Content_Ngrams_3'] else f"""To create a comprehensive guide about the topic, you should focus on the keywords and provide general tips on this topic.
            You should consider the following:
            - The tone should be formal and authoritative.
            - The style should be descriptive and instructional.
            - Use keywords in the headings, subheadings and body paragraphs, and should be easy to understand for human readers.
            - Include information like relevant entities, dates, units, key phrases and lexical relations.
            - Always provide clear instructions about the format (paragraph or list).
            - For list items, give a list definition first.
             - Include the most relevant internal links in the micro context, linking to the best resources that covers the topic.
             - Consider using the following macro semantic words: {", ".join(keywords)}.
              - Good assertions are factual and based on evidence.
             - The average sentence length is: {row['Average_Sentence_Length']} words, it’s better if it remains below 20 words.
              - Use the following types of questions to formulate headings: definitional, grouping, comparative, boolean, representative, represented, and implicit questions.
        """
        return methodology


def main():
  st.title("Semantic Content Brief Creation System")

  # Input fields
  keywords_input = st.text_input("Enter keywords (comma-separated):")
  country_code = st.text_input("Enter country code (e.g., eg, sa, ae, com):", value="com")


  if st.button("Generate Brief"):
      if not keywords_input:
          st.error("Please enter keywords.")
          return

      keywords = [keyword.strip() for keyword in re.split(r',\s*(?!\S)', keywords_input)]
      keywords = [item.strip() for keyword in keywords for item in keyword.splitlines() if item.strip()]

      st.write("Entered keywords:", keywords)

      extractor = ContentExtractor(ua)
      analyzer = TextAnalyzer()
      generator = ReportGenerator()
      try:
          # جلب نتائج البحث من Google وتصفية الروابط
          links = []
          number_of_links = 50

          for keyword in keywords:
              for j in search(keyword, num=number_of_links, stop=number_of_links, pause=1, tld=country_code):
                 links.append(j)

          st.write("Fetched links:", links)
           # تصفية الروابط غير المرغوب فيها
          parsed_urls = []
          for link in links:
            o = urlparse(link)
            url = o.scheme + '://' + o.netloc + o.path
            if('youtube' not in url and 'vimeo' not in url and 'dailymotion' not in url):
                parsed_urls.append(url)

          # إزالة الروابط المكررة
          good_links = list(dict.fromkeys(parsed_urls))
          st.write("Filtered links:", good_links)


          titles, content, headings, lists, tables, meta_descriptions, internal_links, external_links, schema_org_data, open_graph_data, json_ld_data = asyncio.run(extractor.extract_content(good_links))


          content_df = pd.DataFrame({'Content': content, 'query': keywords[0], 'Lists': lists, 'Tables':tables, 'Meta Descriptions':meta_descriptions, 'Internal Links': internal_links, 'External Links':external_links, 'Schema.org':schema_org_data, 'Open Graph':open_graph_data, 'JSON-LD':json_ld_data}) # using only the first keyword for now
          content_df['index'] = content_df.index
          content_df['parsed'] = content_df.Content.apply(nlp)

          # Turn it into a Scattertext corpus
          corpus = (st.CorpusFromParsedDocuments(content_df,
                                               category_col='query',
                                               parsed_col='parsed')
                    .build())
          html = produce_scattertext_explorer(corpus,
                                               category=keywords[0],
                                               category_name=keywords[0],
                                               not_category_name='Other Results',
                                               width_in_pixels=1400,
                                               minimum_term_frequency=2,
                                               term_significance = st.LogOddsRatioUninformativeDirichletPrior())
          open("SERP-Visualization.html", 'wb').write(html.encode('utf-8'))
          st.components.v1.html(html, height=800)

          content_df['Content_Embedding'] = content_df['Content'].apply(analyzer.get_sentence_embedding)
          st.write("DataFrame after content extraction and text embedding:")
          st.dataframe(content_df.head().to_markdown())

          content_df['Content'] = content_df['Content'].fillna("").astype(str)
          content_df['Clean_Content'] = content_df['Content'].str.lower().str.replace('[^\w\s]','')
          content_df['Clean_Content'] = content_df['Clean_Content'].apply(lambda x: [item for item in x.split() if item not in analyzer.stop])
          content_df['Lemmatized_Content'] = content_df['Clean_Content'].apply(lambda x: analyzer.lemmatize_texts([" ".join(x)])[0])
          # تصفية الكلمات المتكررة
          content_df['Lemmatized_Content'] = content_df['Lemmatized_Content'].apply(analyzer.filter_high_frequency_words)
          # تخزين نتائج تحليل spacy مؤقتا
          content_df['Temp_PoS_Tags'] = content_df['Content'].apply(lambda x: analyzer.get_pos_tags_batch([x])[0])
          content_df['Temp_Entities'] = content_df['Content'].apply(lambda x: analyzer.get_entities_batch([x])[0])


          # تصفية الكيانات غير ذات الصلة
          content_df['Entities'] = content_df['Temp_Entities'].apply(lambda x: analyzer.filter_unrelated_entities(x, keywords))

          # تحويل عمود المشاعر إلى عمليات متجهة
          sentiment_results = content_df['Content'].apply(analyzer.analyze_sentiment)
          content_df['Sentiment'], content_df['Sentiment_Score'] = zip(*sentiment_results)
          content_df['Sentiment'], content_df['Sentiment_Score'] = zip(*content_df.apply(lambda row: analyzer.handle_missing_sentiment(row['Sentiment'], row['Sentiment_Score']), axis=1))


          content_df['Average_Sentence_Length'] = content_df['Content'].apply(analyzer.average_sentence_length)


          content_df['Root_Entities'], content_df['Rare_Entities'], content_df['Unique_Entities'] = zip(*content_df['Entities'].apply(lambda x: analyzer.classify_entities(x, keywords)))

          # استخراج العبارات الشائعة من المحتوى
          content_df['Content_Ngrams_3'] = content_df['Lemmatized_Content'].apply(lambda x: analyzer.count_ngrams([x], n=3))
          content_df['Content_Ngrams_4'] = content_df['Lemmatized_Content'].apply(lambda x: analyzer.count_ngrams([x], n=4))
          content_df['Content_Ngrams_5'] = content_df['Lemmatized_Content'].apply(lambda x: analyzer.count_ngrams([x], n=5))
          content_df['Content_Ngrams_6'] = content_df['Lemmatized_Content'].apply(lambda x: analyzer.count_ngrams([x], n=6))

          content_df['Content_Ngrams_3'] = content_df['Content_Ngrams_3'].apply(analyzer.filter_ngrams)
          content_df['Content_Ngrams_4'] = content_df['Content_Ngrams_4'].apply(analyzer.filter_ngrams)
          content_df['Content_Ngrams_5'] = content_df['Content_Ngrams_5'].apply(analyzer.filter_ngrams)
          content_df['Content_Ngrams_6'] = content_df['Content_Ngrams_6'].apply(analyzer.filter_ngrams)

          # إنشاء شبكة الكلمات (Graph)
          content_df['Word_Centrality'] = content_df['Temp_PoS_Tags'].apply(analyzer.create_word_graph)

          #  تعديل عمود Word_Centrality لتخزين الكلمات وقيم المركزية بشكل منفصل
          def split_centrality_values(centrality):
              if centrality:
                words, values = zip(*centrality)
                return list(words), list(values)
              else:
                return [],[]

          content_df['Word'] , content_df['Centrality']= zip(*content_df['Word_Centrality'].apply(split_centrality_values))
          content_df['Semantic_Network'] = content_df['Temp_PoS_Tags'].apply(analyzer.create_semantic_network)
          content_df['Macro_Keywords'], content_df['Micro_Keywords'] = zip(*content_df['Temp_PoS_Tags'].apply(lambda x: analyzer.identify_semantic_words(x, keywords)))
          content_df['Content_Embedding'] = content_df['Content'].apply(analyzer.get_sentence_embedding)


          content_df['Entity_Relationships'] = content_df['Content'].apply(analyzer.analyze_relationships)
          content_df['Text_Summary'] = content_df['Content'].apply(analyzer.summarize_text)
          content_df['Topics'] = content_df['Content'].apply(lambda x: analyzer.get_topics([x]))

          content_df['H1'], content_df['H2'], content_df['Micro_Context_Heading'] = zip(*content_df.apply(lambda row: generator.generate_headings(row, keywords), axis=1))
          content_df['Methodology'] = content_df.apply(lambda row: generator.generate_methodology(row, keywords, country_code), axis=1)


          st.write("DataFrame after text analysis, headings, and methodology generation:")
          st.dataframe(content_df.head().to_markdown())

          # Text generation using GPT-2
          with st.expander("GPT-2 Text Generation"):
              if st.button("Generate Text"):
                  try:
                    generated_text = text_generator(content_df['H1'][0], max_length=100, num_return_sequences=1)
                    st.write("Generated Text:", generated_text[0]['generated_text'])
                  except Exception as e:
                      logging.error(f"Error generating text with gpt2: {e}")
                      st.error("Error during text generation.")

          # Export results to Excel
          output_filename = "output.xlsx"
          with pd.ExcelWriter(output_filename) as writer:
              content_df.to_excel(writer, sheet_name='Content Analysis', index=False)
              pd.DataFrame({'title':titles}).to_excel(writer, sheet_name='Titles', index=False)
              pd.DataFrame({'headings':headings}).to_excel(writer, sheet_name='Headings', index=False)
              st.write(f"\nResults have been exported to {output_filename}")

          # Download Excel file
          with open(output_filename, "rb") as file:
            st.download_button(
                    label="Download Excel",
                    data=file,
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
      except Exception as e:
          logging.error(f"An error has been encountered: {e}")
          st.error(f"An error has been encountered: {e}")

if __name__ == "__main__":
    main()
