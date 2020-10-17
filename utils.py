

def decode_from_tokens(tokens, vocab):
  words = []
  for token in tokens:
    if token.item() == vocab('<eos>'):
        break
    words.append(vocab.idx_to_word(token.item()))
  return ' '.join(words)


def load_texts(path):
  result_dict = {}
  for line in list(open(path, 'r')):
    row = line.split('\t')
    idx = int(row[0])
    sentence = row[1].strip()
    if idx in result_dict:
      result_dict[idx].append(sentence)
    else:
      result_dict[idx] = [sentence]