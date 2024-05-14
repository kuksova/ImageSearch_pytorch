# ImageSearchApp
An app that helps the user find specific images with text that match specific text keyword.

# Problem Statement
The problem I am trying to solve is the difficulty in efficiently organizing and retrieving specific screenshots from the large archive of photos, particularly those containing text such as names or other relevant information. It is a challenge to manually search through this extensive collection to find screenshots related to specific topics, individuals, or text content. The proposed solution aims to address this problem by providing a tool that allows the user to search for screenshots based on text queries and automatically organize their screenshot collection accordingly.

# Demo 
![](https://github.com/kuksova/ImageSearchApp/blob/main/demo/demo.gif)

# Usage 
You can run the demo GUI via main_demo.py
```
$ python main_demo.py
```
Or you can the demo without GUI via  main_search.py
Required Arguments: imgs_folder, query_text
```
$ python main_search.py
```


# Pipeline  
On process

# The technology stack:
- Pure Pytorch without aditional API 
- Text detection model CRAFT Craft_mlt_25k.pth (81 MB) (pretrained weigths on SynthText, IC13, IC1 ) -> quntized_Craft (20 MB)
- Text recognition model CRNN+VGG+CTCLossConverter Cyrillic_g2.pth (15 MB) 
- Sqlite3 Database 


# Metrics
Text recognition CPU time 5 sec per image. 
Search in database 0.007 sec.

