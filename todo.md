# Preprocessing Todos

1. Should replace URLs with something generic. This is what BERT tokenizer does to URLs:

```py
   ['http', ':', '/', '/', 'home', '.', 'hampton',
    '##ro', '##ads', '.', 'com', '/', 'stories', '/',
    'story', '.', 'cf', '##m', '?', 'story', '=', '105',
    '##52', '##2', '&', 'ran', '=', '48', '##57', '##7', '`', '`'
```

2. Same for XML tags:

```py
    ['<', '$', 'blog', '##back', '##link', '##au', '##thor', '$', '>', '@', '<', '$', 'blog', '##back',
     '##link', '##date', '##time', '$', '>']
```

3. Email Addresses as well. 