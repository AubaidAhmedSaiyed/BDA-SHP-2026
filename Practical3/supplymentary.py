STOP_WORDS = {"a", "an", "the", "is", "in", "on", "and", "of", "to", "for", "with"}
 
for token in tokens:
    if token and token not in STOP_WORDS:
        print(f"{token}\t1")
