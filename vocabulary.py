#!/usr/bin/env python
"""Defines the Vocabulary class
"""

class Vocabulary(object):
    def __init__(self, lowercase=True):
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}
        self.nwords = 0
        self.lowercase = lowercase

    @classmethod
    def from_word2idx_dict(cls, word2idx_dict, lowercase=True):
        instance = cls(lowercase)
        instance.word2idx = word2idx_dict
        for k, v in word2idx_dict.items():
            instance.idx2word[v] = k
        instance.nwords = len(word2idx_dict)
        instance.lowercase = lowercase
        return instance

    def __call__(self, word):
        """
        Returns the id corresponding to the word
        """
        w = word.lower() if self.lowercase else word
        return self.word2idx['<unk>'] if w not in self.word2idx else self.word2idx[w]

    def __len__(self):
        """
        Get the number of words in the vocabulary
        """
        return self.nwords
