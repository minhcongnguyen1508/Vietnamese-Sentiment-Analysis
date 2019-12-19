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
        s = re.sub('[!@#$%^&*()_+-=><:;?/\|`~*"]', '', s)
        result.append(s)
    return result

def RemoveDuplicate(x):
    result = []
    for s in x:
        s = re.sub(r'([a-z])\1+', lambda m: m.group(1), s, flags=re.IGNORECASE)
        s = re.sub(r'([a-z][a-z])\1+', lambda m: m.group(1), s, flags=re.IGNORECASE)
        result.append(s)
    return result
