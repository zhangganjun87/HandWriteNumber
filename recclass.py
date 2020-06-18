#
import tensorflow as tf
import cv2
import numpy as np
import keras.backend  as K
from PIL import Image
import sys
import os,io
import os
from tensorflow.python.platform import gfile
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class Recdctc(object):
    def __init__(self):
        return
    
    def load_model(self,):
        from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D
        from keras.models import Model
        from densenet import dense_cnn
        self.g2 = tf.Graph()
        self.sess2 = tf.Session(graph=self.g2 )
        with self.sess2.as_default():
           with self.g2.as_default():
            input = Input(shape=(32, None, 1), name='the_input')
            nclass = 5991
            y_pred = dense_cnn(input, nclass)
            self.basemodel = Model(inputs=input, outputs=y_pred)
			#---------------------test--------------------------------
            jsonFile = self.basemodel.to_json()
            with open(r'./test_modle.json', 'w') as file:
                file.write(jsonFile)
			#---------------------------------------------------------	
            modelPath = r'weights-densent-13.hdf5'#手写体数字5991
            self.basemodel.load_weights(modelPath)
            print('xxxxxxxxxxxxxxxxxxxxxxx')
            return self.basemodel

    def load_the_dict_from_txt(self,):
        char = ''
        import io
        with io.open('char.txt', 'r', encoding='utf-8') as f:
            for ch in f.readlines():
                # print ('ch= '+ch)
                ch = ch.strip('\r\n')
                char = char + ch
        char = char[1:] + '卍'
        # print(char)
        nclass = len(char)
        #print('nclass:', len(char))
        id_to_char = {i: j for i, j in enumerate(char)}
        f.close()
        return id_to_char, nclass
    
    def predict_pic(self,src):
        id_to_char, nclass = self.load_the_dict_from_txt()
        img_x = src
        h_, w_ = img_x.shape[:2]
        # src, top, bottom, left, right, borderType
        top = 0
        bottom = 0
        left = 15 # 15
        if (w_ < 320):
            right = 320 - w_  # 320 - w_ #320 - w_  # 250-w_#320-w_#不pad
        else:
            right = 0 # 15#15#15
        borderType = cv2.BORDER_CONSTANT
        # VVV_b = VVV_g = VVV_r =int(np.mean(img_x[0][0:100]))#int(np.mean(img_x[0][0:100]))
        img = cv2.copyMakeBorder(img_x, top, bottom, left, right, borderType, value=[235, 255, 255])
        # cv2.imshow('ccc',img)
        # #cv2.waitKey()
        # pad
        # bgr-rgb
        #img = cv2pil(img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        im = img.convert('L')
        scale = im.size[1] * 1.0 / 32
        w = im.size[0] / scale
        w = int(w)
        # print('w:',w)
        im = im.resize((w, 32), Image.ANTIALIAS)
        img = np.array(im).astype(np.float32) / 255.0 - 0.5
        X = img.reshape((32, w, 1))
        X = np.array([X])
        # t.tic()
        with self.sess2.as_default():
            with self.sess2.graph.as_default():
                y_pred = self.basemodel.predict(X)
                # t.toc()
                # print("y_pred,",y_pred)
                argmax = np.argmax(y_pred, axis=2)[0]
                y_pred = y_pred[:, :, :]

                char_list = []
                #argmax = pred_text
                for i in range(len(argmax)):
                    if argmax[i] != nclass - 1 and ((not (i > 0 and argmax[i] == argmax[i - 1])) or (
                            i > 1 and argmax[i] == argmax[i - 2])):
                        char_list.append(id_to_char[argmax[i]])
                # print('char_list:',char_list)
                char_list = [x for x in char_list if x != '卍']
                print('结果是2:', ''.join(char_list))
                out=''.join(char_list)
                del y_pred
                del X
                import gc
                gc.collect()
                #K.clear_session()
                return out


#工具方法：把.pb文件生成.pbtxt文件
def convert_pb_to_pbtxt(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', 'model.pbtxt', as_text=True)
    return


#工具方法：查看.pb文件中各个节点的名称
def getPbNodeName(pbFileName):
    with tf.gfile.FastGFile(pbFileName, 'rb') as f:
        graph_def = tf.GraphDef()    #定义一个Graph
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='') #把graph_def作为当前默认的图
        
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    with open('./nodeName.txt', 'w+') as fw:
        for tensor_name in tensor_name_list:
            fw.write(tensor_name + '\n')
            
#工具方法：用tensorflow调用.pb模型，判断.pb模型的效果
def predict_tf(inputFile, pbFile):
    id_to_char, nclass = load_the_dict_from_txt_1()
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        
        with open(pbFile, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name='')
            
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            graph = tf.get_default_graph()    #3.获得当前图
            
            x = graph.get_tensor_by_name('the_input:0')
            y_predict = graph.get_tensor_by_name('out/truediv:0')
            
            img_x = cv2.imread(inputFile)
            h_, w_ = img_x.shape[:2]
            top = 0
            bottom = 0
            left = 15 # 15
            if (w_ < 320):
                right = 320 - w_  # 320 - w_ #320 - w_  # 250-w_#320-w_#不pad
            else:
                right = 0 # 15#15#15
            borderType = cv2.BORDER_CONSTANT
            img = cv2.copyMakeBorder(img_x, top, bottom, left, right, borderType, value=[235, 255, 255])
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            im = img.convert('L')
            scale = im.size[1] * 1.0 / 32
            w = im.size[0] / scale
            w = int(w)
            # print('w:',w)
            im = im.resize((w, 32), Image.ANTIALIAS)
            img = np.array(im).astype(np.float32) / 255.0 - 0.5
            X = img.reshape((32, w, 1))
            X = np.array([X])
            
            y_pred = sess.run(y_predict, feed_dict={x:X})
            
            argmax = np.argmax(y_pred, axis=2)[0]
            y_pred = y_pred[:, :, :]

            char_list = []
            #argmax = pred_text
            for i in range(len(argmax)):
                if argmax[i] != nclass - 1 and ((not (i > 0 and argmax[i] == argmax[i - 1])) or (
                        i > 1 and argmax[i] == argmax[i - 2])):
                    char_list.append(id_to_char[argmax[i]])
            # print('char_list:',char_list)
            char_list = [x for x in char_list if x != '卍']
            print('结果是2:', ''.join(char_list))
            
            
#加载字典，生成相应List
def load_the_dict_from_txt_1():
    char = ''
    import io
    with io.open('char.txt', 'r', encoding='utf-8') as f:
        for ch in f.readlines():
            # print ('ch= '+ch)
            ch = ch.strip('\r\n')
            char = char + ch
    char = char[1:] + '卍'
    # print(char)
    nclass = len(char)
    #print('nclass:', len(char))
    id_to_char = {i: j for i, j in enumerate(char)}
    f.close()
    return id_to_char, nclass


#使用dnn加载.pb模型进行预测
def useDnnModel(srcImg):
    #1.前处理
    id_to_char, nclass = load_the_dict_from_txt_1()
    img_x = srcImg
    h_, w_ = img_x.shape[:2]
    top = 0
    bottom = 0
    left = 15 # 15
    if (w_ < 320):
        right = 320 - w_  # 320 - w_ #320 - w_  # 250-w_#320-w_#不pad
    else:
        right = 0 # 15#15#15
    borderType = cv2.BORDER_CONSTANT
    img = cv2.copyMakeBorder(img_x, top, bottom, left, right, borderType, value=[235, 255, 255])
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    im = img.convert('L')
    scale = im.size[1] * 1.0 / 32
    w = im.size[0] / scale
    w = int(w)
    # print('w:',w)
    im = im.resize((w, 32), Image.ANTIALIAS)
    img = np.array(im).astype(np.float32) / 255.0 - 0.5
    X = img.reshape((32, w, 1))
    X = np.array([X])
    
    #2.导入模型
    hwNumNet = cv2.dnn.readNetFromTensorflow('./model.pb', './model.pbtxt')   #
    hwNumNet.setInput(X)
    y_pred = hwNumNet.forward()
    

    #3.后处理
    argmax = np.argmax(y_pred, axis=2)[0]
    y_pred = y_pred[:, :, :]
    char_list = []
    #argmax = pred_text
    for i in range(len(argmax)):
        if argmax[i] != nclass - 1 and ((not (i > 0 and argmax[i] == argmax[i - 1])) or (
                i > 1 and argmax[i] == argmax[i - 2])):
            char_list.append(id_to_char[argmax[i]])
    # print('char_list:',char_list)
    char_list = [x for x in char_list if x != '卍']
    print('结果是2:', ''.join(char_list))
    out=''.join(char_list)
    

#use optimize_for_inference
def optimizePb(pbFilePath):
    from tensorflow.python.tools import optimize_for_inference_lib
    from tensorflow.tools.graph_transforms import TransformGraph

    with tf.gfile.FastGFile(pbFilePath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ['Placeholder'], ['final_result'], tf.float32.as_datatype_enum)
        graph_def = TransformGraph(graph_def, ['module_apply_default/hub_input/Sub'], ['final_result'], ['remove_nodes(op=PlaceholderWithDefault)', 'strip_unused_nodes(type=float, shape=\"1,224,224,3\")', 'sort_by_execution_order'])
        with tf.gfile.FastGFile('./inference_graph.pb', 'wb') as f:
            f.write(graph_def.SerializeToString())






# if __name__=='__main__':
      # rec_0 = Recdctc()  # 识别身份证号和日期
      # rec_0.load_model()
       # dst_2 = cv2.imread(r'roi.jpg')
      # # out_all = rec_0.predict_pic(dst_2)
      
       # useDnnModel(dst_2)   #使用cv2.dnn加载.pb模型
     
     
     # convert_pb_to_pbtxt('./model.pb')  #根据已有的.pb模型生成.pbtxt模型
     
     # getPbNodeName('./model.pb')   #显示.pb模型中各个节点中的名称
     
     # predict_tf(r'roi.jpg', './model.pb')   #tensorflow调用.pb模型进行预测
     
     # optimizePb(r'./model.pb')
     
     
     
     

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

from keras import backend as K

K.set_learning_phase(0)  # all new operations will be in test mode from now on

sess = K.get_session()

model = load_model("liveness2.model")

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

# inputs:  ['conv2d_1_input']
print('inputs: ', [input.op.name for input in model.inputs])

# outputs:  ['activation_6/Softmax']
print('outputs: ', [output.op.name for output in model.outputs])


tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)
tf.train.write_graph(frozen_graph, "model", "tf_model.pbtxt", as_text=True)





     