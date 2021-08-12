from collections import Counter


class Tokenizer:

    def __init__(self, translations):
        self.translations = translations
        self.unk = -1

    def build_dict(self):
        text = ''
        for t in self.translations:
            text += t

        char_counts = Counter(text)
        sorted_vocab = sorted(char_counts, key=char_counts.get, reverse=True)
        int_to_char = {ii: word for ii, word in enumerate(sorted_vocab, 1)}

        char_to_int = {word: ii for ii, word in int_to_char.items()}

        return int_to_char, char_to_int

    def encode(self, sent, char_to_int):

        encoded = []
        char_list = list(sent)
        for c in char_list:
            try:
                encoded.append(char_to_int[c])

            except KeyError:
                encoded.append(self.unk)
        return encoded

    def decode_text(self, encoded_chars, int_to_char):

        decoded = ''
        for e in encoded_chars:
            try:
                decoded += int_to_char[e]

            except KeyError:
                decoded += ''

        return decoded
