### Extract text using command line tools

#### xmlstarlet

```bash

sudo apt install xmlstarlet

```

Extract space delimited text:

```bash
xmlstarlet sel -t -m "//w" -c "text()" -o " " sou_1945_1.xml
```

Extract space delimited text for specific part-of-speech types:

```bash
xmlstarlet sel -t -m "//w[@pos='NN']" -c "text()" -o " " sou_1945_1.xml
```

