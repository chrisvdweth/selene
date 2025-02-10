import string

from nltk import word_tokenize
from nltk import pos_tag

from nltk.corpus import stopwords

nltk_stopwords = list(stopwords.words('english'))
punctuation_translator = str.maketrans('', '', string.punctuation)


def remove_punctuation(s):
    return s.translate(punctuation_translator)



def preprocess_text(s, tokenizer=None, remove_stopwords=True, remove_punctuation=True, 
                    stemmer=None, lemmatizer=None, lowercase=True, return_type='str'):
    # Throw an error if both stemmer and lemmatizer are not None
    if stemmer is not None and lemmatizer is not None:
         raise ValueError("Stemmer and Lemmatizer cannot both be not None!")
    
    # Tokenization either with default tokenizer or user-specified tokenizer
    if tokenizer is None:
        token_list = word_tokenize(s)
    else:
        token_list = tokenizer.tokenize(s)

    # Stem or lemmatize if needed
    if lemmatizer is not None:
        token_list = lemmatize_token_list(lemmatizer, token_list)
    elif stemmer is not None:
        token_list = stem_token_list(stemmer, token_list)
    
    # Convert all tokens to lowercase if need
    if lowercase:
        token_list = [ token.lower() for token in token_list ]
    
    # Remove all stopwords if needed
    if remove_stopwords:
        token_list = [ token for token in token_list if not token in nltk_stopwords ]
        
    # Remove all punctuation marks if needed (note: also converts, e.g, "Mr." to "Mr")
    if remove_punctuation:
        token_list = [ ''.join(c for c in s if c not in string.punctuation) for s in token_list ]
        token_list = [ token for token in token_list if len(token) > 0 ] # Remove "empty" tokens
    
    if return_type == 'list':
        return token_list
    elif return_type == 'set':
        return set(token_list)
    else:
        return ' '.join(token_list)
    

    
def stem_token_list(stemmer, token_list):
    for idx, token in enumerate(token_list):
        token_list[idx] = stemmer.stem(token)
    return token_list
    
    
def lemmatize_token_list(lemmatizer, token_list):
    pos_tag_list = pos_tag(token_list)
    for idx, (token, tag) in enumerate(pos_tag_list):
        tag_simple = tag[0].lower() # Converts, e.g., "VBD" to "c"
        if tag_simple in ['n', 'v', 'j']:
            word_type = tag_simple.replace('j', 'a') 
        else:
            word_type = 'n'
        lemmatized_token = lemmatizer.lemmatize(token, pos=word_type)
        token_list[idx] = lemmatized_token
    return token_list
