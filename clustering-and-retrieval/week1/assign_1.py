import graphlab
import matplotlib.pyplot as plt
import numpy as np
from sets import Set

"""
Get a table of the most frequent words in the given person's wikipedia page.
"""
def top_words(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False)

def has_words(unique_words):
    def is_subset(word_count_vector):
        return unique_words.issubset(Set(word_count_vector.keys()))
    return is_subset

def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)

def compute_length(row):
    return len(row['text'])



wiki = graphlab.SFrame('people_wiki.gl')

wiki['word_count'] = graphlab.text_analytics.count_words(wiki['text'])


model = graphlab.nearest_neighbors.create(wiki, label='name', 
    features=['word_count'], method='brute_force', distance='euclidean')

model.query(wiki[wiki['name']=='Barack Obama'], label='name', k=10)



obama_words = top_words('Barack Obama')
obama_words


barrio_words = top_words('Francisco Barrio')
barrio_words

bush_words = top_words('George W. Bush')



obama_words.join(barrio_words, on='word')
combined_words = obama_words.join(barrio_words, on='word')

combined_words


combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})
combined_words


combined_words.sort('Obama', ascending=False)


obama_common_words = Set(['the', 'in', 'and', 'of', 'to'])



wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)




obama = wiki[wiki['name']=='Barack Obama']
barrio = wiki[wiki['name']=='Francisco Barrio']
biden = wiki[wiki['name']=='Joe Biden']  
bush = wiki[wiki['name']=='George W. Bush']
lawrence = wiki[wiki['name']=='Lawrence Summers']  
romney = wiki[wiki['name']=='Mitt Romney']   



has_obama_words = beautifyl_currying(obama_common_words)
has_zoeira_words = beautifyl_currying(Set(['bigorna', 'lerere']))

has_obama_words(romney['word_count'])
has_zoeira_words(romney['word_count'])



wiki['has_top_words'] = wiki['word_count'].apply(has_obama_words)

wiki[0]['word_count']





# wiki possui 59071 palavras
# numero de documentos que possuem 
#   as mesmas palavras do obama: 56066







print 'Output from your function:', has_obama_words(wiki[32]['word_count'])
print 'Output from your function:', has_obama_words(wiki[33]['word_count'])





graphlab.distances.euclidean(obama['word_count'][0], bush['word_count'][0])
# 34.3

graphlab.distances.euclidean(obama['word_count'][0], biden['word_count'][0])
# 33.0

graphlab.distances.euclidean(bush['word_count'][0], biden['word_count'][0])
# 32.7




obama_bush = obama_words.join(bush_words, on='word')
obama_bush = obama_bush.rename({'count':'Obama', 'count.1':'Bush'})
obama_bush.sort('Obama', ascending=False)




# tf-idf now.
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['word_count'])


model_tf_idf = graphlab.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
    method='brute_force', distance='euclidean')




model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)





obama_tf_idf = top_words_tf_idf('Barack Obama')
obama_tf_idf



schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
schiliro_tf_idf




obama_schiliro = obama_tf_idf.join(schiliro_tf_idf, on='word')
obama_schiliro = obama_schiliro.rename({'weight':'Obama', 'weight.1':'Schiliro'})
obama_schiliro




has_obama_tf_idf_words = has_words(Set('obama,law,democratic,senate,presidential'.split(',')))

has_obama_tf_idf_words(romney['tf_idf'])


wiki['same_obama'] = wiki['tf_idf'].apply(has_obama_tf_idf_words)




graphlab.distances.euclidean(obama['tf_idf'][0], biden['tf_idf'][0])




wiki['length'] = wiki.apply(compute_length) 




nearest_neighbors_euclidean = model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=100)
nearest_neighbors_euclidean = nearest_neighbors_euclidean.join(wiki[['name', 'length']], on={'reference_label':'name'})



nearest_neighbors_euclidean.sort('rank')



plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([1000, 5500, 0, 0.004])
plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size':16})
plt.tight_layout()



nearest_neighbors_cosine = model2_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=100)
nearest_neighbors_cosine = nearest_neighbors_cosine.join(wiki[['name', 'length']], on={'reference_label':'name'})



nearest_neighbors_cosine.sort('rank')