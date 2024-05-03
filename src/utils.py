

def trim_to_shortest(text1, text2):
    shorter = min(len(text1.split(' ')), len(text2.split(' ')))
    text1 = " ".join(text1.split(' ')[:shorter])
    text2 = " ".join(text2.split(' ')[:shorter])
    return text1, text2

