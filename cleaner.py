text = open("f_vision.py", encoding= "utf-8").read()
text = text.replace("\xa0", " ")
open("f_vision.py", "w", encoding= "utf-8").write(text)
print("File was cleaned")