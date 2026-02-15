This is an API for question generation. A list of commands can be found under QUICK_START.md

# Basic Idea
The api takes in input data (currently SAT questions) and extracts metadata + skeletons from that data. It then takes that metadata, feeds it into an AI model (currently Gemini for cost efficiency) and then creates questions. 

# What's different
This is more than just prompting GPT for questions after feeding it a few samples. Here's why:
- question lengths, answer types, etc. all preserved; the model (based on the type and difficulty of question) generates a question of appropriate length
- topics/skeletons are preserved. In other words, the base & format of each question is rooted in the way previous questions are worded

Advantages: faster question gen, allows for adaptive questions to be created, along with all the other advantages an AI model has over humans

Issues: 
- questions are not innately original. Making them completely original will stray away from keeping the question grounded and reflective of what it's meant to be (observed in early prototyping of the api)- lose innate human touch
- all questions will not have their own flair/challenge that a human will add. It's just looking at and rewording existing questions. As a direct result, this'll only be effective if a large corpus of previous questions are available. If the corpus is too small (which it currently is - sitting at only 1600 questions), the questions will be of lower quality (as is expected)

More updates planned soon:
- fewer logic skeletons for standardized question gen (can tune the desired amount of each question type - 20% true/false, 80% mcq for example)
- larger corpus (probably going to look for other datasets of public questions, like math competition problems for example - which btw is a huge problem since students always want more practice and companies sell questions for ridiculous prices)
- public release (no need to build from source)
