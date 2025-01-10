import re

query = 'Штрафы на курсе Pythom abs.c'

words = re.split(r'[. ]+', query)

tokenized_query = [word.lower() for word in words if word]

print(tokenized_query)