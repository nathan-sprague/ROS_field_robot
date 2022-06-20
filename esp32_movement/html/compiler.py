

f = open("index.html", "r")

allOneWord=""
for x in f:
  line = ""
  
  a=0
  commented=False
  lastChar = ""
  for y in x:
    if commented==False or y=="\n":
      if y=="\\":
         allOneWord+="\\\\"
      elif y=="\n":
        if lastChar!="\n":
           allOneWord+="\\n"
           commented=False
      # elif (lastChar==" " or lastChar=="\t") and (y==" " or y=="\t"):
      #   pass
      elif y=='"':
        allOneWord+='\\"'
      elif y=="//":
        commented=True
      else:
        allOneWord+=y
      lastChar=y
    
print(allOneWord)


  

print("done")


