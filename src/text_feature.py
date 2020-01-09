import unidecode
import string
import os
import re

neg_emoticons = {':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
                      '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😪', '😥', '😓', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
                      '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<', 
                      ':-[', ':[', ':{'}

pos_emoticons = {'=))', ':v', ';)', '^^', '<3', '☀', '☺', '♡', '♥', '✌', '✨', '❣', '❤', '🌝', '😍', '😊', '😂', '😆', '😅',
                     '🌷', '🌸', '😎', '5*', '😜', '😘', '😁', '❣️', '😉', '😁', '😇', '⭐', '🤣', '✨', '👏🏻', '🌹', '😙','💥',
                      '🌺', '🌼', '🍓', '💯', '🎈', '🐅', '🐶', '🐾', '👉', '👌', '👍', '👍🏻', '👏', '👻', '💃', '💄', '💋',
                      '💌', '💎', '💐', '💓', '💕', '💖', '💗', '💙', '💚', '💛', '💜', '❤', '💞', ':-)', ':)', ':D', ':o)',
                      ':]', ':3', ':c)', ':>', '=]', '8)'}
emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
def LowerCase(x):
    # x is List review
    return [s.lower() for s in x]

def CountEmoticons(sentence):
    pos_count = 0
    neg_count = 0
    for emoticon in pos_emoticons:
        pos_count += sentence.count(emoticon)
    for emoticon in neg_emoticons:
        neg_count += sentence.count(emoticon)
    return pos_count, neg_count

def RemoveEmoticons(x):
    result = []
    for sentence in x:
        for emoticon in pos_emoticons:
            sentence = sentence.replace(emoticon, ' positive ')
        for emoticon in neg_emoticons:
            sentence = sentence.replace(emoticon, ' negative ')
        result.append(sentence)
    return result

def RemoveTone(x):
    result = []
    for s in x:
        s = re.sub(emoj, ' ', s)
        s = re.sub('[!@#$%^&*()_+-=><:;?/\|`~*"]', ' ', s)
        result.append(s)
    return result

def RemoveDuplicate(x):
    result = []
    for s in x:
        s = re.sub(r'([a-z])\1+', lambda m: m.group(1), s, flags=re.IGNORECASE)
        s = re.sub(r'([a-z][a-z])\1+', lambda m: m.group(1), s, flags=re.IGNORECASE)
        s = re.sub(r'( )\1+', lambda m: m.group(1), s, flags=re.IGNORECASE)
        result.append(s)
    return result
