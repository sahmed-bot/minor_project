from django.shortcuts import render


def covid(request):
    import numpy as np
    import tensorflow.compat.v1 as tf
    import os, argparse
    import cv2
    tf.disable_v2_behavior()
  
    #tf.disable_v2_behavior()
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weightspath = os.path.join(BASE_DIR,'ml_model','models')
    metaname = 'model.meta'
    ckptname = 'model-8485'
    #imagepath=os.path.join(os.path.dirname('~/models'),'images','ex-covid.jpeg')
    imagepath = 'C:/Users/Coding/Covid/COVID-Net/assets/ex-covid.jpeg'
    
    parser = argparse.ArgumentParser(description='COVID-Net Inference')
    parser.add_argument('--weightspath', default='./models/COVIDNet-CXR-Large', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model-8485', type=str, help='Name of model ckpts')
    parser.add_argument('--imagepath', default='./COVID-Net/assets/ex-covid.jpeg', type=str, help='Full path to image to be inferenced')
    
    #args = parser.parse_args()
    
    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
    
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
      saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))
      saver.restore(sess, os.path.join(weightspath, ckptname))
    
      image_tensor = graph.get_tensor_by_name("input_1:0")
      pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")
    
      x = cv2.imread(imagepath)
      h, w, c = x.shape
      x = x[int(h/6):, :]
      x = cv2.resize(x, (224, 224))
      x = x.astype('float32') / 255.0
      pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

    
      #print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
      #print('Confidence')
      #print('Normal: {:.3f}, Pneumonia: {:.3f}, COVID-19: {:.3f}'.format(pred[0][0], pred[0][1], pred[0][2]))
      results={}
      results['N']='{:.3f}'.format(pred[0][0])#normal
      results['P']='{:.3f}'.format(pred[0][1])#pneumonia
      results['C']='{:.3f}'.format(pred[0][2])#covid
      #results['N']=pred[0][0]#normal
      #results['P']=pred[0][1]#pneumonia
      #results['C']=pred[0][2]#covid

    return render (request, 'index.html',results)
     # print('**DISCLAIMER**')
      #print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')
