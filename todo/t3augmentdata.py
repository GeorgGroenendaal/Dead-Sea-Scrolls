import pickle


sentences_to_read = open("sentences.pickle", "rb")
sentences = pickle.load(sentences_to_read)
sentences_to_read.close()
#print(sentences)

images_to_read = open("images.pickle", "rb")
images = pickle.load(images_to_read)
images_to_read.close()
print(images)