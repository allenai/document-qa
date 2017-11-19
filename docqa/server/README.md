## Server
This contains our code to run the demo server. It uses [bing search](https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/)
 to located web documents related to the input question, and/or [TAGME](https://tagme.d4science.org/tagme/) to 
 locate relevant Wikipedia documents. Both services requires API keys, TAGME is free but Bing charges a small
 fee for each search. 
 
 Running this code requires some additional dependencies, they can be installed with
 
 `pip install -r docqa/server/requirements.txt`
 
 docqa/server/qa_system.py contains the end-to-end question answering system that can use these services to answer questions.
 
 docqa/server/server.py contains the code to run the server. 
 
 