from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import facenet
import pickle


def classify_face(sess,graph,images_placeholder,embeddings,phase_train_placeholder,embedding_size):
    img_paths = ["aligned/somename/tmp.png"]
    pickle_file = "../../model/my_model.pkl"
    # model = "../../pretrained_model/"


    with graph.as_default():


        print("loaded model")

        # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        # embedding_size = embeddings.get_shape()[1]

        emb_array = np.zeros((1, embedding_size))

        start_index = 0
        end_index = 1

        images = facenet.load_data(img_paths, False, False, 160)
        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
        file = open("mine.txt", "wb")
        file.write(emb_array.tobytes())

        with open(pickle_file, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        print('Loaded classifier model from file "%s"' % pickle_file)

        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            # print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
            return (class_names[best_class_indices[i]],best_class_probabilities)
        # accuracy = np.mean(np.equal(best_class_indices, labels))
        # print('Accuracy: %.3f' % accuracy)


# classify_face()