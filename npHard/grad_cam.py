import numpy as np
import cv2
from keras import backend as K
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.nasnet import NASNetMobile
import tensorflow as tf
from tensorflow.python.framework import ops
import datafr
# Define model here ---------------------------------------------------
def build_model(string):
    """Function returning keras model instance.
    
    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """
    if(string==1):
        return ResNet50(include_top=True, weights='imagenet')
    elif(string==0):
        return VGG16(include_top=True, weights='imagenet')

    elif (string==2):
        return NASNetMobile(include_top=True, weights='imagenet')
    elif (string==3):
        return MobileNet(include_top=True, weights='imagenet')

H, W = datafr.H, datafr.W # Input shape, defined by the model (model.input_shape)
# ---------------------------------------------------------------------

def load_image(path,number=2,preprocess=True):
    """Load and preprocess image."""
    x = path
    #######
    #print('----------------')
    #print(x)
    if preprocess:
        x = image.img_to_array(x)
        #print(x)
        x = np.expand_dims(x, axis=0)
        if(number==1):
            from keras.applications.resnet50 import ResNet50, preprocess_input
            x = preprocess_input(x)
        elif(number==0):
            from keras.applications.vgg16 import VGG16, preprocess_input
            x = preprocess_input(x)
        elif (number==2):
            from keras.applications.nasnet import NASNetMobile, preprocess_input
            x = preprocess_input(x)
        elif (number==3):
            from keras.applications.mobilenet import MobileNet, preprocess_input
            x = preprocess_input(x)
        #print(x)
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def build_guided_model(string):
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model(string)
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (datafr.H, datafr.W), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])    
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    
    # Process CAMs
    new_cams = np.empty((images.shape[0], datafr.H, datafr.W))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()
    return new_cams




##def dicto(model, guided_model, img_path, layer_name='res5c_branch2c', cls=-1, visualize=True, save=True,string=0,knob=5):
##    """Compute saliency using all three approaches.
##        -layer_name: layer to compute gradients;
##        -cls: class number to localize (-1 for most probable class).
##    
##    """
##    
##    if(string==1):
##        from keras.applications.resnet50 import decode_predictions
##        datafr.loader+=10
##    elif(string==0):
##        from keras.applications.vgg16 import decode_predictions
##        datafr.loader+=10
##
##    elif (string==2):
##        from keras.applications.nasnet import decode_predictions
##        datafr.loader+=10
##    elif (string==3):
##        from keras.applications.mobilenet import decode_predictions
##        datafr.loader+=10
##
##
##
##    string=string
##    
##    preprocessed_input = load_image(img_path,number=string)
##    datafr.loader+=10
##    predictions = model.predict(preprocessed_input)
##    datafr.loader+=10
##    top_n = int(knob)
##    top = decode_predictions(predictions, top=top_n)[0]
##    classes = np.argsort(predictions[0])[-top_n:][::-1]
##    #print('Model prediction:')
##    dictvariables={}
##    for c, p in zip(classes, top):
####        if datafr.flag==1:
####            datafr.d[p[1]]=[c,p[2]]
####        if datafr.flags1==1:
####            datafr.d1[p[1]]=[c,p[2]]
##        dictvariables[p[1]]=[c,p[2]]
##        
##        print('\t{:15s}\t({})\twith probability {:.3f}'.format(p[1], c, p[2]))
##        return dictvariables
##



def compute_saliency(model, guided_model, img_path, layer_name='res5c_branch2c', cls=-1, visualize=True, save=True,string=0,knob=5):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    
    """
    
    if(string==1):
        from keras.applications.resnet50 import decode_predictions
        datafr.loader+=10
    elif(string==0):
        from keras.applications.vgg16 import decode_predictions
        datafr.loader+=10

    elif (string==2):
        from keras.applications.nasnet import decode_predictions
        datafr.loader+=10
    elif (string==3):
        from keras.applications.mobilenet import decode_predictions
        datafr.loader+=10



    string=string
    
    preprocessed_input = load_image(img_path,number=string)
    datafr.loader+=10
    predictions = model.predict(preprocessed_input)
    datafr.loader+=10
    top_n = int(knob)
    top = decode_predictions(predictions, top=top_n)[0]
    classes = np.argsort(predictions[0])[-top_n:][::-1]
    #print('Model prediction:')
    dictvariables={}
    for c, p in zip(classes, top):
##        if datafr.flag==1:
##            datafr.d[p[1]]=[c,p[2]]
##        if datafr.flags1==1:
##            datafr.d1[p[1]]=[c,p[2]]
        dictvariables[p[1]]=[c,p[2]]
        
        print('\t{:15s}\t({})\twith probability {:.3f}'.format(p[1], c, p[2]))
    datafr.flag=0
    datafr.loader+=10
    if cls == -1:
        cls = np.argmax(predictions)
    #class_name = decode_predictions(np.eye(1, 1000, cls))[0][0][1]
    #print("Explanation for '{}'".format(class_name))
    
    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
    datafr.loader+=10
    gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    datafr.loader+=10
    guided_gradcam = gb * gradcam[..., np.newaxis]

    datafr.loader+=10
    if save:
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + load_image(img_path,number=string, preprocess=False)) / 2
        cv2.imwrite('assets/gradcam.jpg', np.uint8(jetcam))
        print("Yess....\n\n")
        try:
            cv2.imwrite('assets/guided_backprop.jpg', deprocess_image(gb[0]))
            print("Yess....\n\n")
            cv2.imwrite('assets/guided_gradcam.jpg', deprocess_image(guided_gradcam[0]))
        except Exception as e:
            print(e)
    K.clear_session()
    datafr.loader+=10
    return gradcam, gb, guided_gradcam,dictvariables

#if __name__ == '__main__':
    #model = build_model()
    #guided_model = build_guided_model()
    #gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, layer_name='block5_conv3',
                                             #img_path=sys.argv[1], cls=-1, visualize=False, save=True)
