# Contextual Chatbot 
This is an AI Contextual Chatbot built in PyTorch using the concepts of Natural Language Processing and Neural Networks. 
   * The implementation is straightforward with a Feed Forward Neural net with 3 hidden layers. 
   * You can customize it as per your requirement by just modifying **intents.json** with your own tags and responses and re-running the training process.
   
The approach is inspired by this article and implemented in PyTorch: [Article](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077)

## Description <h3>
   * **nltk_utils.py**
   
      This file contains the functions used to prepare the pre-processing pipeline which involves:
      * **Tokenization**
      * **Stemming**
      * Creating a **Bag of Words**
   * **model.py**
   
      This file contains the Feed Forward Neural net with **4 layers** (3 are hidden layers).
   * **train.py**
      
      This file involves:
      * Creating training data using **Dataset, DataLoader** from **torch.utils.data**
      * Training process on our Neural Net from **model.py** 
      * Saving the trained model to **data.pth** 
   * **chat.py**
   
      This file involves:
      * Loading of the saved model
      * Pre-processing of the user given inputs
      * Predicting the **tag** and printing the corresponding **response**
      
   * **intents.json**
      
      This json file contains the tags - labels, patterns - inputs(unprocessed) and responses that the bot will use to respond to given user inputs. 
      This file contains 9 tags namely: "greeting", "goodbye", "thanks", "name", "age", "work", "hobbies", "play" and "live".
      
 
 ## Example
 An example of working of the bot:
 
 ![ChatBot Example](https://i.ibb.co/C65mwDF/Project.png)
 
 ## Customize
 Let's take a look at **intents.json**. You can customize it according to your requirement. Just define a new tag, possible patterns, and possible responses for the chat bot. You have to re-run the training whenever this file is modified.

 ![intents.json](https://i.ibb.co/GtVZsMg/Project-1.png)
