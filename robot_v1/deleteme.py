import regex as re
p = re.compile(r'hello (\w*)')
print(p.sub(r'goodbye \1', 'well hello ece'))