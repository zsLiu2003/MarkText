str_text = "1 2 3 4 5,,, 6 7  7   8"
out = ""
for cur in str_text:
    if cur == "\u0020":
        cur = "\u2004"
    
    out = out + cur
print("111\u0020111")
print(out)