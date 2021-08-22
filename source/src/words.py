from random import randrange

class WordSource:
    
    words = None

    def __init__(self) -> None:
        file = open('wordlist.txt', 'r')
        self.words = file.readlines()
        file.close()

    def get_random_word(self):
        
        while True:
            num = randrange(1000)
            curr_word = self.words[num]
            
            if('j' not in curr_word and 'z' not in curr_word and 'a' not in curr_word and 'n' not in curr_word):
                return curr_word[:-1]


