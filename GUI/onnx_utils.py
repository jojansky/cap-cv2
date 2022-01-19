import onnxruntime
import numpy as np
import cv2
from sklearn.decomposition import PCA

# declare global variables
data_dir = 'datasets'
mod_dir = 'models'
onnx_model = '{}/best.onnx'.format(mod_dir)
onnx_labels = '{}/labels.txt'.format(mod_dir)

def run_model():
    # Run the model on the backend
    session = onnxruntime.InferenceSession(onnx_model, None)

    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name  

    return input_name, session

def load_labels(path):
    # read from a .txt file
    list_data = []
    with open(path) as f:
        data = f.read().split('\n')  
        #print("data: %s" %(data))
        list_data.extend(data)
    return np.asarray(list_data)

def postprocess(results):
    pred_result = results[0]
    #print ("pred_result:", pred_result)
    list_prob = np.delete(results, 0, 0)
    dict_prob = list_prob.reshape(-1)[0]
    #for k in dict_prob.keys():
    #    print("Probablities for %s is %.4f" %(k, dict_prob[k]))
    return np.array(list(dict_prob.values()))

""" So the logic goes like this. 
1. We need to save the image that we get, use a random number for it.
2. We need to process the image
3. Then we need to save the processed image, give the src to the exit that is it.
What else can we do. 
"""

def classify_car (img_file,uuid):
    img = compress_pca(img_file)
    img = cv2.imread(img_file)    
    img = cv2.resize(img,(224,224))
    img = img.astype(np.float32)
    img = img.reshape(1, 3, 224, 224)


    # img contains the array. We need to 
    # Preprocessing the image

    input_name, session = run_model()
    labels = load_labels(onnx_labels)   
    raw_result = session.run([], {input_name: img})
    print ("Raw Result Length", len(raw_result))
    print ("Raw Result Length", len(raw_result[0]))
    for e in raw_result:
        print ("Length ", len(e))
        print ("type of e", type(e))

    print (raw_result[1][0])
    print (len(raw_result[1][0]) )

    # abcd = cv2.convertScaleAbs(raw_result[1][0],alpha=(255.0))
    # cv2.imwrite('D:\\code\\carclassification\\output\\whatever.jpg',abcd)

    # print ("Shape",np.shape(raw_result))
    #list_results = postprocess(raw_result)
    #print("res of postprocess:", len(list_results))

    img_file = ""
    return img_file

def classify_car1(img_file):
    """ To classify the car """

    input_name, session = run_model()
    #print('Input Name:', input_name)
    labels = load_labels(onnx_labels)
    #print("Labels:", type(labels), labels.shape, len(labels), labels)

    #img_file = sg.PopupGetFile('Please enter a file name')
    #print("Image size: ", image.size)
    print("Image: ", img_file)

    # image normalization
    image = Image.open(img_file).resize((224, 224), Image.LANCZOS)
    image_data = np.array(image).transpose(2, 0, 1)
    norm_img_data = cv2.normalize(image_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    input_data = norm_img_data.reshape(1, 3, 224, 224)

    #start = time.time()
    raw_result = session.run([], {input_name: input_data})
    #end = time.time()
    print(raw_result)
    # list_results = postprocess(raw_result)
    #print("res of postprocess:", len(list_results), list_results)

    sort_idx = np.flip(np.squeeze(np.argsort(list_results)))
    list_labels = labels[sort_idx[:5]]
    list_prob = list_results[sort_idx[:5]]

    fig2_name = plot_bar_prob(list_labels, list_prob, 'Bird classification', 'Top 5')

    inference_time = np.round((end - start) * 1000, 2)
    print('Inference time: ' + str(inference_time) + " ms")

    return fig2_name



### Below function to do PCA transformation of images to reduce dimesionality. 
### INPUT - image and n_components (default is 50)
### OUTPUT - compressed images returned.
def compress_pca(image,PCA_n_component=50):
    img = cv2.imread(image,cv2.COLOR_BGR2RGB)
    # img = image
    try:
        if(len(img.shape) < 3):
            print(len(img.shape))
            print("\n file_path: ",image)
            gray = cv2.imread(image,0)
            blue=red=green=gray
        else:
            red,green,blue = cv2.split(img)
    except:
        print("Exception encountered")

    try:
        df_blue = blue/255
        df_green = green/255
        df_red = red/255
        pca_b = PCA(n_components=PCA_n_component)
        pca_b.fit(df_blue)
        trans_pca_b = pca_b.transform(df_blue)
        pca_g = PCA(n_components=PCA_n_component)
        pca_g.fit(df_green)
        trans_pca_g = pca_g.transform(df_green)
        pca_r = PCA(n_components=PCA_n_component)
        pca_r.fit(df_red)
        trans_pca_r = pca_r.transform(df_red)
        b_arr = pca_b.inverse_transform(trans_pca_b)
        g_arr = pca_g.inverse_transform(trans_pca_g)
        r_arr = pca_r.inverse_transform(trans_pca_r)
        b_arr1 = b_arr.astype(np.float)
        g_arr1 = g_arr.astype(np.float)
        r_arr1 = r_arr.astype(np.float)
        img_reduced= (cv2.merge((r_arr1, g_arr1, b_arr1)))

    except:
        return None
    img_reduced = cv2.convertScaleAbs(img_reduced,alpha=(255.0))
    if not cv2.imwrite(image, img_reduced):
        raise Exception("Could not write image")

    return img_reduced