#!/bin/bash -v

pip install nltk gensim pandas simplejson spacy

# download and install Mallet
cd && wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip && unzip mallet-2.0.8.zip
# git clone https://github.com/mimno/Mallet.git ~/

exit 0
