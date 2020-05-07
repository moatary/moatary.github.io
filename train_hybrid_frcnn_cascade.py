from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model, load_model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

if 'tensorflow' == K.backend():
    import tensorflow.compat.v1 as tf  # import tensorflow as tf

    tf.disable_v2_behavior()

from keras.backend.tensorflow_backend import set_session

config2 = tf.ConfigProto()
config2.gpu_options.allow_growth = True
set_session(tf.Session(config=config2))

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.",
                  default=10)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='vgg')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=true).",
                  action="store_false", default=True)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90",
                  help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=16)
parser.add_option("--config_filename", dest="config_filename", help=
"Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.",
                  default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path",
                  help="Input path for weights. If not specified, will try to load default weights provided by keras.",
                  default="C:\\Users\\Apadana\\Desktop\\moattari\\keras_frcnn\\models\\resnet50\\voc - Copy.hdf5")
parser.add_option("--rpn", dest="rpn_weight_path", help="Input path for rpn.", default=None)
parser.add_option("--opt", dest="optimizers", help="set the optimizer to use", default="SGD")
parser.add_option("--elen", dest="epoch_length", help="set the epoch length. def=1000", default=1000)
parser.add_option("--load", dest="load", help="What model to load", default=None) #resnet50#TODO : iN THE TEST PHASE, CHANGE TO "voc"
parser.add_option("--dataset", dest="dataset", help="name of the dataset", default="voc")
parser.add_option("--cat", dest="cat", help="categroy to train on. default train on all cats.", default=None)
parser.add_option("--lr", dest="lr", help="learn rate", type=float, default=1e-3)

(options, args) = parser.parse_args()

# if not options.train_path:   # if filename is not given
#     parser.error('Error: path to training data must be specified. Pass --path to command line')
if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

# mkdir to save models.
if not os.path.isdir("models"):
    os.mkdir("models")
if not os.path.isdir("models/" + options.network):
    os.mkdir(os.path.join("models", options.network))
C.model_path = os.path.join("models", options.network, options.dataset + ".hdf5")
C.num_rois = int(options.num_rois)

# we will use resnet. may change to others
if options.network == 'vgg' or options.network == 'vgg16':
    C.network = 'vgg16'
    from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
    from keras_frcnn import resnet as nn

    C.network = 'resnet50'
elif options.network == 'vgg19':
    from keras_frcnn import vgg19 as nn

    C.network = 'vgg19'
elif options.network == 'mobilenetv1':
    from keras_frcnn import mobilenetv1 as nn

    C.network = 'mobilenetv1'
elif options.network == 'mobilenetv2':
    from keras_frcnn import mobilenetv2 as nn

    C.network = 'mobilenetv2'
elif options.network == 'densenet':
    from keras_frcnn import densenet as nn

    C.network = 'densenet'
else:
    print('Not a valid model')
    raise ValueError

# check if weight path was passed via command line
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    # set the path to weights based on backend and model
    C.base_net_weights = nn.get_weight_path()

# all_imgs, classes_count, class_mapping = get_data(options.train_path, options.cat)
all_imgs, classes_count, class_mapping = pickle.load(open('leishmania_frcnndata2.pkl', 'rb'))
class_mapping['parasites']=2
class_mapping['bg']=0
# get_data(options.train_path, options.cat)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length,
                                               K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,
                                             K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# optimizer setup
if options.optimizers == "SGD":
    if options.rpn_weight_path is not None:
        optimizer = SGD(lr=options.lr / 100, decay=0.0009, momentum=0.9)
        optimizer_classifier = SGD(lr=options.lr / 5, decay=0.0005, momentum=0.9)
    else:
        optimizer = SGD(lr=options.lr / 10, decay=0.0005, momentum=0.9)
        optimizer_classifier = SGD(lr=options.lr / 10, decay=0.0005, momentum=0.9)
else:
    optimizer = Adam(lr=options.lr, clipnorm=0.001)
    optimizer_classifier = Adam(lr=options.lr, clipnorm=0.001)

# may use this to resume from rpn models or previous training. specify either rpn or frcnn model to load
if options.load is not None:
    print("loading previous model from ", options.load)
    model_rpn.load_weights(options.load, by_name=True)
    model_classifier.load_weights(options.load, by_name=True)
elif options.rpn_weight_path is not None:
    print("loading RPN weights from ", options.rpn_weight_path)
    model_rpn.load_weights(options.rpn_weight_path, by_name=True)
else:
    print("no previous model was loaded")

# compile the model AFTER loading weights!
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count) - 1)],
                         metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = int(options.epoch_length)
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

for epoch_num in range(num_epochs):
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    # first 3 epoch is warmup
    if epoch_num == 3 and options.rpn_weight_path is not None:
        K.set_value(model_rpn.optimizer.lr, options.lr / 30)
        K.set_value(model_classifier.optimizer.lr, options.lr / 3)

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            X, Y, img_data = next(data_gen_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.4,
                                       max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, 0] == 1)
            pos_samples = np.where(Y1[0, :, 0] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois // 2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()
                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3])),
                            ("average number of objects", len(selected_pos_samples))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)

                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('Training complete, exiting.')







class preprocess_images():
    import numpy as np
    def __init__(self,trainpath=None):
        ## Parameters to learn
        #AFTER GRAYSCALING and thresholding# define avged size of the biggest compartment ## FOR RESCALING
        self.baseline_segsize=None
        # after grayscaling and otsu # for rescaling test images
        self.baseline_density_sqroot=None
        # define baseline histogram list ## for global equalization
        self.baseline_histogram_R=None
        self.baseline_histogram_G=None
        self.baseline_histogram_B=None
        # define standard minimum/maximum intensities
        self.baseline_min_R = None
        self.baseline_min_G = None
        self.baseline_min_B = None
        self.baseline_max_R = None
        self.baseline_max_G = None
        self.baseline_max_B = None
        #
        if trainpath is not None:
            self.trainpath=trainpath
        else:
            self.trainpath=None
        ## set default values based on pretrain
        self.meanOverAll_maxs_per_chan = [248.83108108, 250.2027027, 247.51351351]
        self.meanOverAll_mins_per_chan = [0.46621622, 0. , 4.97972973]
        self.mean_scale_metric = 33.18292292676569



    def histogram(self,image):
        if len(image.shape)==2:
            image=(image/np.max(image)*255).astype(np.int8)
            imm=np.asarray([np.reshape(image,[-1])]*np.prod(image.shape)).T == np.array([list(range(255))]*np.prod(image.shape))
            hist=np.sum(imm,0)
        else:
            hist=[]
            for itm in range(image.shape[2]):
                image=(image[:,:,itm]/np.max(image[:,:,itm])*255).astype(np.int8)
                imm=np.asarray([np.reshape(image,[-1])]*np.prod(image.shape)).T == np.array([list(range(255))]*np.prod(image.shape))
                hist.append(np.sum(imm,0))
        return hist

    def tmp1(self,image):
        return preprocess_images.scale_function(image)

    def tmp2(self,image):
        return scale_function(image)

    def tmp3(self,image):
        return scale_function(image)

    def tmp4(self,image):
        return scale_function2(image)

    def scale_function(self,im):
        im = np.dot(im, [0.2989, 0.5870, 0.1140])
        # first smooth and binary thresh the image uSING THE PREVIOUSLY SOUGHT THRESH BELOW
        image2 = denoise_image(im)
        # pass to thresholding
        image3 = threshold_image(image2)
        image3 = (image3 > 0).astype(int)
        # then use diff and maximize over positive values
        diff = np.diff(image3, axis=1)
        # find positive locations (start of a segment)
        poslocs = np.where(diff > 0)
        poslocs = poslocs[0] * 10000 + poslocs[1]
        anynonzeros = np.where(diff < 0)
        anynonzeros = anynonzeros[0] * 10000 + anynonzeros[1]
        try:
            rep1 = np.array([poslocs] * len(anynonzeros))
            rep2 = np.array([anynonzeros] * len(poslocs)).T
            dist = rep2 - rep1
            dist[dist < 0] = 0
            dist[dist == 0] = 100000
            minn = np.min(dist, 1)
            minn = minn[minn < 100]
            # np.delete(minn,np.where(minn>400))
            mean_metric = np.mean(minn)
            return mean_metric
        except:
            return -1

    @staticmethod
    def scale_function2(im):
        im = np.dot(im, [0.2989, 0.5870, 0.1140])
        # first smooth and binary thresh the image uSING THE PREVIOUSLY SOUGHT THRESH BELOW
        image2 = denoise_image(im)
        # pass to thresholding
        image3 = threshold_image(image2)
        image3 = (image3 > 0).astype(int)
        # then use diff and maximize over positive values
        diff = np.diff(image3, axis=1)
        # find positive locations (start of a segment)
        poslocs = np.where(diff > 0)
        poslocs = poslocs[0] * 10000 + poslocs[1]
        anynonzeros = np.where(diff < 0)
        anynonzeros = anynonzeros[0] * 10000 + anynonzeros[1]
        try:
            rep1 = np.array([poslocs] * len(anynonzeros))
            rep2 = np.array([anynonzeros] * len(poslocs)).T
            dist = rep2 - rep1
            dist[dist < 0] = 0
            dist[dist == 0] = 100000
            minn = np.min(dist, 1)
            minn = minn[minn < 100]
            # np.delete(minn,np.where(minn>400))
            mean_metric = np.mean(minn)
            return mean_metric
        except:
            return -1


    def fit(self,trainpath=None,filetype='jpg'):
        import os
        from skimage import transform
        from skimage import io
        import skimage as sk
        if trainpath is None:
            trainpath= self.trainpath
        listimages= os.listdir(self.trainpath)
        listimages=[itm for itm in listimages if itm.split('.')[-1]=='jpg']
        ## PREREQUISITES
        _IMAGE_THRESH = 0.5
        #
        ##
        self.maxs_per_chan= []
        self.mins_per_chan= []
        self.meanOverAll_maxs_per_chan=np.array([0,0,0])
        self.meanOverAll_mins_per_chan=np.array([0,0,0])
        self.all_scale_metrics_train = []
        self.mean_scale_metric = 0


        ##
        ### second pass all training images through the histogram equalization, then find maximums, minimums, maxcontinuouslen, and density
        for ii,imp in enumerate(listimages):
            print('fitting to %d percent of %d images.'%(ii/len(listimages)*100,len(listimages)))
            im=io.imread(os.path.join(trainpath, imp))
            im=im.astype(np.int32)
            im0,im1,im2= im[:,:,0],im[:,:,1],im[:,:,2]
            self.meanOverAll_maxs_per_chan = (ii*self.meanOverAll_maxs_per_chan +np.array([np.max(im0),np.max(im1),np.max(im2)]))/(ii+1)
            self.meanOverAll_mins_per_chan = (ii*self.meanOverAll_mins_per_chan +np.array([np.min(im0),np.min(im1),np.min(im2)]))/(ii+1)
            ### get maxcontinuouslen
            #self.all_scale_metrics_train.append(scale_function(im))
            scaleval=scale_function(im)
            if scaleval!=0:
                self.mean_scale_metric= (ii*self.mean_scale_metric + scaleval)/(ii+1)



        # from skimage.io import imread, imsave,imshow
        # img= imread('C:\\Users\\m3.aaaa\\Desktop\\jahad\\projects2\\leishmania__detection\\Leishmania\\GoldStandardsDrBahreyni\\20190525_132818.jpg')
        # image= denoise_image(img)
        # return image


    def transform(self,image=None,imagepaths=None,multi_channel=True, path2save='temp'):
        from skimage.io import imread, imshow,imsave
        from skimage.transform import rescale
        if image is not None:
            if isinstance(image,str):
                image= imread(image)
            ## modify image intensities for each channel
            # get image intensity
            if multi_channel==True:
                # print('here22')

                min_intensities_perchannel = np.min(np.min(image,0),0)
                max_intensities_perchannel = np.max(np.max(image,0),0)
                # print('here222')
                minchange= self.meanOverAll_mins_per_chan-min_intensities_perchannel
                maxchange= self.meanOverAll_maxs_per_chan-max_intensities_perchannel
                # print('here2222')
                # transform to averaged intensity
                m= maxchange/(minchange+0.001)
                # print('here22222')
                image[:, :, 0] = (image[:, :, 0] - min_intensities_perchannel[0])*m[0] + self.meanOverAll_mins_per_chan[0]
                image[:, :, 1] = (image[:, :, 1] - min_intensities_perchannel[1])*m[1] + self.meanOverAll_mins_per_chan[1]
                image[:, :, 2] = (image[:, :, 2] - min_intensities_perchannel[2])*m[2] + self.meanOverAll_mins_per_chan[2]
                # print('here222222')
                #
                ## rescale the image to make parasites sizes the same
                scalemetr = self.scale_function2(image)
                # print('here2222222')
                scal= self.mean_scale_metric / scalemetr
                if scal<1.06:
                    image = rescale(image, scal, anti_aliasing=False)
                return image
            else:
                min_intensity = np.min(np.min(image,0),0)
                max_intensity = np.max(np.max(image,0),0)
                minperchan= np.dot(self.meanOverAll_mins_per_chan, [0.2989, 0.5870, 0.1140])
                maxperchan= np.dot(self.meanOverAll_maxs_per_chan, [0.2989, 0.5870, 0.1140])
                minchange= minperchan - min_intensity
                maxchange= maxperchan - max_intensity
                m= maxchange/(minchange+0.001)
                # transform to averaged intensity
                image = (image - min_intensity)*m + minperchan
                ## rescale the image to make parasites sizes the same
                scalemetr = self.scale_function2(image)
                image = rescale(image, self.mean_scale_metric/scalemetr, anti_aliasing=False)
                return image
        else:
            import os
            try:
                os.mkdir(path2save)
            except:
                pass
            allpaths_results=[]
            for ii, impath in enumerate(imagepaths):
                print('performing image number%d/%d'%(ii+1,len(imagepaths)))
                ## modify image intensities for each channel
                image= imread(impath)
                # get image intensity
                if multi_channel==True:
                    min_intensities_perchannel = np.min(np.min(image,0),0)
                    max_intensities_perchannel = np.max(np.max(image,0),0)
                    minchange= self.meanOverAll_mins_per_chan-min_intensities_perchannel
                    maxchange= self.meanOverAll_maxs_per_chan-max_intensities_perchannel
                    # transform to averaged intensity
                    m= maxchange/(minchange+0.001)
                    image[:, :, 0] = (image[:, :, 0] - min_intensities_perchannel[0])*m[0] + self.meanOverAll_mins_per_chan[0]
                    image[:, :, 1] = (image[:, :, 1] - min_intensities_perchannel[1])*m[1] + self.meanOverAll_mins_per_chan[1]
                    image[:, :, 2] = (image[:, :, 2] - min_intensities_perchannel[2])*m[2] + self.meanOverAll_mins_per_chan[2]
                    #
                    ## rescale the image to make parasites sizes the same
                    scalemetr = self.scale_function2(image)
                    image = rescale(image, self.mean_scale_metric/scalemetr, anti_aliasing=False)
                else:
                    min_intensity = np.min(np.min(image,0),0)
                    max_intensity = np.max(np.max(image,0),0)
                    minperchan= np.dot(self.meanOverAll_mins_per_chan, [0.2989, 0.5870, 0.1140])
                    maxperchan= np.dot(self.meanOverAll_maxs_per_chan, [0.2989, 0.5870, 0.1140])
                    minchange= minperchan - min_intensity
                    maxchange= maxperchan - max_intensity
                    m= maxchange/(minchange+0.001)
                    # transform to averaged intensity
                    image = (image - min_intensity)*m + minperchan
                    ## rescale the image to make parasites sizes the same
                    scalemetr = self.scale_function2(image)
                    image = rescale(image, self.mean_scale_metric/scalemetr, anti_aliasing=False)
                imagename=os.path.join(path2save, impath.split('\\')[-1])
                imsave(imagename, image)
                allpaths_results.append(imagename)







def getMacrophagesParasitesImagesRects():
    import numpy
    import pickle
    from matplotlib import pyplot
    import numpy as np
    from skimage import io
    import pandas as pd
    # get (SINGLE_PARASITES) imagepaths
    xls = pd.ExcelFile("D:\\pprsproj\\leishmania\\all.xls")
    sheetX = xls.parse(0)  # 2 is the sheet number
    var1 = sheetX['filenames']
    imagpaths = var1.values
    imagpaths_parasites= [itm[:-1].replace('C:\\Users\\m3.aaaa\\Desktop\\jahad\\projects2\\leishmania__detection\\Leishmania\\GoldStandardsDrBahreyni\\', 'C:\\Users\\Apadana\\Desktop\\moattari\\cascade_test\\GoldStandardsDrBahreyni\\') for itm in imagpaths]
    # repair path
    for ii in range(len(imagpaths_parasites)):
        imagpaths_parasites[ii]=imagpaths_parasites[ii].replace('\'F:\\DOCS\\M Sc\\Cooperation\\moattari\\leishmania\\all\\','C:\\Users\\Apadana\\Documents\\MATLAB\\Leishmania\\all\\')
        imagpaths_parasites[ii]=imagpaths_parasites[ii].replace('\'C:\\Users\\Apadana\\Documents\\MATLAB\\Leishmania\\all\\','C:\\Users\\Apadana\\Documents\\MATLAB\\Leishmania\\all\\')
    # get (SINGLE_PARASITES) rectangles
    sheetX = xls.parse(1)  # 2 is the sheet number
    var1,var2,var3,var4 = sheetX['a'],sheetX['b'],sheetX['c'],sheetX['d']
    var1,var2,var3,var4 = var1.values,var2.values,var3.values,var4.values

    Ns = np.asarray([io.imread(itm).shape[1] for itm in imagpaths_parasites ])
    imagrects= np.array([var1.tolist(),var2.tolist(),var3.tolist(),var4.tolist()]).T
    imagrects_parasites= imagrects.copy()
    imagrects_parasites[:, 2]= imagrects[:, 0] + imagrects[:, 2] - 1
    imagrects_parasites[:, 1] =  Ns - imagrects[:, 1] - imagrects[:, 3]
    imagrects_parasites[:, 3] = Ns - imagrects[:, 1]
    imagrects_parasites=imagrects_parasites.astype(int)
    #
    #
    # get (MACROPHAGES) imagepaths
    xls = pd.ExcelFile("D:\\pprsproj\\leishmania\\all2.xlsx")
    sheetX = xls.parse(0)  # 2 is the sheet number
    var1 = sheetX['filenames']
    imagpaths = var1.values
    imagpaths_macrophages= [itm[:-1].replace('C:\\Users\\m3.aaaa\\Desktop\\jahad\\projects2\\leishmania__detection\\Leishmania\\GoldStandardsDrBahreyni\\', 'C:\\Users\\Apadana\\Desktop\\moattari\\cascade_test\\GoldStandardsDrBahreyni\\') for itm in imagpaths]
    # repair path
    for ii in range(len(imagpaths_macrophages)):
        imagpaths_macrophages[ii]=imagpaths_macrophages[ii].replace('\'F:\\DOCS\\M Sc\\Cooperation\\moattari\\leishmania\\all\\','C:\\Users\\Apadana\\Desktop\\moattari\\cascade_test\\GoldStandardsDrBahreyni\\')
        imagpaths_macrophages[ii]=imagpaths_macrophages[ii].replace('\'C:\\Users\\Apadana\\Documents\\MATLAB\\Leishmania\\all\\','C:\\Users\\Apadana\\Desktop\\moattari\\cascade_test\\GoldStandardsDrBahreyni\\')
    # get (MACROPHAGES) rectangles
    sheetX = xls.parse(1)  # 2 is the sheet number
    var1,var2,var3,var4 = sheetX['a'],sheetX['b'],sheetX['c'],sheetX['d']
    var1,var2,var3,var4 = var1.values,var2.values,var3.values,var4.values
    imagrects= np.array([var1.tolist(),var2.tolist(),var3.tolist(),var4.tolist()]).T
    imagrects_macrophages=imagrects.copy()
    Ns = np.asarray([io.imread(itm).shape[1] for itm in imagpaths_macrophages ])
    x1 = Ns - imagrects_macrophages[:, 1]
    x2 = Ns - (imagrects_macrophages[:, 1] + imagrects_macrophages[:, 3])
    y1 = imagrects_macrophages[:, 0]
    y2 = (imagrects_macrophages[:, 0] + imagrects_macrophages[:, 2])
    # crop the rect from it
    imagrects_macrophages[:, 0],imagrects_macrophages[:, 1],imagrects_macrophages[:, 2],imagrects_macrophages[:, 3] = np.min(np.array([y1, y2]),0), np.min(np.array([x1, x2]),0), np.max(np.array([y1, y2]),0), np.max(np.array([x1, x2]),0)
    imagrects_macrophages=imagrects_macrophages.astype(int)
    #
    return imagpaths_parasites,imagrects_parasites,imagpaths_macrophages,imagrects_macrophages


def leishmaniadata2rcnn():
    ## presepecs
    # train-test percent
    TRAINPERCENT = 0.75

    ## : debug
    # path to images and rects
    import numpy as np
    from skimage import io
    imagpaths_parasites, imagrects_parasites, imagpaths_macrophages, imagrects_macrophages= getMacrophagesParasitesImagesRects()
    #
    #
    ##Parasites
    allimgs=[]
    allimgsnames=[]
    allimgsids=[]
    classes_count = {}
    class_mapping = {'parasites':0, 'macrophages':1, 'bg':2}

    # iterate over all files and generate main syntax
    jj=-1
    for ii,(impath,rect) in enumerate(zip(imagpaths_parasites, imagrects_parasites)):
        class_name ='parasites'
        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        if ii==0:
            lastpath=thispath= impath
            append2bbox = True
            im = io.imread(impath)
            h, w = im.shape[:2]
            annotation_data = {'filepath': impath, 'width': w, 'height': h, 'bboxes': [{'class': 'parasites', 'x1': rect[1], 'x2': rect[3], 'y1': rect[0], 'y2': rect[2], 'difficult': 1}]}
            # allimgsnames.append(impath)
            # allimgsids.append(jj)
            ##TODO: make it right/. TRAINDATA +VALIDATIONDATA
            _randomizer = np.random.random()
            if _randomizer<TRAINPERCENT:
                # use it for trainval data
                annotation_data['imageset'] = 'trainval'
            else:
                # use it for test data
                annotation_data['imageset'] = 'test'

            # if element_filename in trainval_files:
            #     annotation_data['imageset'] = 'trainval'
            # elif element_filename in test_files:
            #     annotation_data['imageset'] = 'test'
            # else:
            #     annotation_data['imageset'] = 'trainval'

        else:
            thispath=impath
            if thispath==lastpath:
                append2bbox=True
                annotation_data['bboxes'].append({'class': 'parasites', 'x1': rect[1], 'x2': rect[3], 'y1': rect[0], 'y2': rect[2], 'difficult': 1})
            else:
                lastpath= thispath
                jj+=1
                allimgsnames.append(impath)
                allimgsids.append(jj)
                allimgs.append(annotation_data)
                im = io.imread(impath)
                h, w = im.shape[:2]
                annotation_data = {'filepath': impath, 'width': w, 'height': h, 'bboxes': [{'class': 'parasites', 'x1': rect[1], 'x2': rect[3], 'y1': rect[0], 'y2': rect[2], 'difficult': 1}] }
                _randomizer = np.random.random()
                if _randomizer < TRAINPERCENT:
                    # use it for trainval data
                    annotation_data['imageset'] = 'trainval'
                else:
                    # use it for test data
                    annotation_data['imageset'] = 'test'
    #
    #
    ##macrophages
    # iterate over all files and generate main syntax
    for ii,(impath,rect) in enumerate(zip(imagpaths_macrophages, imagrects_macrophages)):
        # find the occurrence of impath in imagpaths_parasites
        class_name ='macrophages'
        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        try:
            tmp_dictimgs_paths = [allimgs[itm]['filepath'] for itm in range(len(allimgs))]
            desired_id=tmp_dictimgs_paths.index(impath)
        except:
            desired_id=len(allimgs)
        thispath=impath
        ##TODO: make it right/. TRAINDATA +VALIDATIONDATA
        # if element_filename in trainval_files:
        #     annotation_data['imageset'] = 'trainval'
        # elif element_filename in test_files:
        #     annotation_data['imageset'] = 'test'
        # else:
        #     annotation_data['imageset'] = 'trainval'
        if desired_id<len(allimgsids):
            allimgs[desired_id]['bboxes'].append({'class': 'macrophages', 'x1': rect[1], 'x2': rect[3], 'y1': rect[0], 'y2': rect[2], 'difficult': 1}) ##TODO: what is 'difficult'
        else:
            jj+=1
            im = io.imread(impath)
            h, w = im.shape[:2]
            annotation_data = {'filepath': impath, 'width': w, 'height': h, 'bboxes': [{'class': 'macrophages', 'x1': rect[1], 'x2': rect[3], 'y1': rect[0], 'y2': rect[2], 'difficult': 1}]}
            allimgsnames.append(impath)
            allimgsids.append(jj)
            _randomizer = np.random.random()
            if _randomizer<TRAINPERCENT:
                # use it for trainval data
                annotation_data['imageset'] = 'trainval'
            else:
                # use it for test data
                annotation_data['imageset'] = 'test'
            allimgs.append(annotation_data)
            #
    # # select image background for each unified item: 'bg'
    # for ii,img in enumerate(allimgs):
    #     # find all occupied locs in image
    #
    #     # randomize background until

    # add background images of both macrophages and parasites to the list as new images with full region
    classes_count['bg'] = 0
    import os
    from skimage import io
    #
    path='C:\\Users\\Apadana\\Documents\\MATLAB\\Leishmania\\Parasite\\notParasite\\'
    lst=os.listdir(path)
    lst=[path+itm for itm in lst]
    for ii,impath in enumerate(lst):
        classes_count['bg']=classes_count['bg']+1
        im=io.imread(impath)
        sz=im.shape
        tmp={'filepath': impath, 'width': sz[1], 'height': sz[0], 'bboxes': [{'class': 'bg', 'x1': 0, 'x2': sz[1]-1, 'y1': 0, 'y2': sz[0]-1, 'difficult': 1}]}
        _randomizer = np.random.random()
        if _randomizer < TRAINPERCENT:
            # use it for trainval data
            tmp['imageset'] = 'trainval'
        else:
            # use it for test data
            tmp['imageset'] = 'test'
        #
        allimgs.append(tmp)
    #
    path='D:\\pprsproj\\Leishmani_matlab\\Parasite\\sought_fps\\results_parasiteDetector_accurate_butremovestoomuch_0_15\\'
    lst=os.listdir(path)
    lst=[path+itm for itm in lst]
    path='D:\\pprsproj\\Leishmani_matlab\\Parasite\\sought_fps\\results1\\'
    lst2=os.listdir(path)
    lst2=[path+itm for itm in lst2]
    lst.extend(lst2)
    path='D:\\pprsproj\\Leishmani_matlab\\Parasite\\sought_fps\\DUPLICATE2\\'
    lst2=os.listdir(path)
    lst2=[path+itm for itm in lst2]
    lst.extend(lst2)
    for ii,impath in enumerate(lst):
        classes_count['bg']=classes_count['bg']+1
        im=io.imread(impath)
        sz=im.shape
        tmp={'filepath': impath, 'width': sz[1], 'height': sz[0], 'bboxes': [{'class': 'bg', 'x1': 0, 'x2': sz[1]-1, 'y1': 0, 'y2': sz[0]-1, 'difficult': 1}]}
        _randomizer = np.random.random()
        if _randomizer < TRAINPERCENT:
            # use it for trainval data
            tmp['imageset'] = 'trainval'
        else:
            # use it for test data
            tmp['imageset'] = 'test'
        #
        allimgs.append(tmp)
    #
    path='C:\\Users\\Apadana\\Documents\\MATLAB\\Leishmania\\NegativeTemplates\\all\\'
    lst=os.listdir(path)
    lst=[path+itm for itm in lst]
    for ii,impath in enumerate(lst):
        classes_count['bg']=classes_count['bg']+1
        im=io.imread(impath)
        sz=im.shape
        tmp={'filepath': impath, 'width': sz[1], 'height': sz[0], 'bboxes': [{'class': 'bg', 'x1': 0, 'x2': sz[1]-1, 'y1': 0, 'y2': sz[0]-1, 'difficult': 1}]}
        _randomizer = np.random.random()
        if _randomizer < TRAINPERCENT:
            # use it for trainval data
            tmp['imageset'] = 'trainval'
        else:
            # use it for test data
            tmp['imageset'] = 'test'
        #
        allimgs.append(tmp)
    #
    path='D:\\pprsproj\\Leishmani_matlab\\Macrophage\\sought_fps\\results_pruned_metrent\\'
    lst=os.listdir(path)
    lst=[path+itm for itm in lst]
    for ii,impath in enumerate(lst):
        classes_count['bg']=classes_count['bg']+1
        im=io.imread(impath)
        sz=im.shape
        tmp={'filepath': impath, 'width': sz[1], 'height': sz[0], 'bboxes': [{'class': 'bg', 'x1': 0, 'x2': sz[1]-1, 'y1': 0, 'y2': sz[0]-1, 'difficult': 1}]}
        tmp={'filepath': impath, 'width': sz[1], 'height': sz[0], 'bboxes': [{'class': 'bg', 'x1': 0, 'x2': sz[1]-1, 'y1': 0, 'y2': sz[0]-1, 'difficult': 1}]}
        _randomizer = np.random.random()
        if _randomizer < TRAINPERCENT:
            # use it for trainval data
            tmp['imageset'] = 'trainval'
        else:
            # use it for test data
            tmp['imageset'] = 'test'
        #
        allimgs.append(tmp)



    return allimgs, classes_count, class_mapping


def matlabrects2pythonrects(file='leishmania_frcnndata.pkl'):
    import pickle
    all_imgs, classes_count, class_mapping= pickle.load(open(file, 'rb'))
    for ii,itm in enumerate(all_imgs):
        for jj,bbx in enumerate(itm['bboxes']):
            itm['bboxes'][jj]['x1']=bbx['x1']-1
            itm['bboxes'][jj]['x2']=bbx['x2']-1
            itm['bboxes'][jj]['y1']=bbx['y1']-1
            itm['bboxes'][jj]['y2']=bbx['y2']-1
        all_imgs[ii]=itm
    pickle.dump([all_imgs, classes_count, class_mapping],open(file,'wb'))







def denoise_image(img):
    # define and import
    # from skimage.io import imread,imsave
    # img=imread('C:\\Users\\m3.aaaa\\Desktop\\jahad\\projects2\\leishmania__detection\\Leishmania\\GoldStandardsDrBahreyni\\20190525_132818.jpg')
    import skimage as sk
    import skimage.color as clr
    import numpy as np
    from skimage.morphology import erosion, dilation, opening, closing
    from skimage.morphology import disk
    from skimage import transform
    import scipy.ndimage as ndi
    from needed_functions import comparetunes, show_images_labels, comparetunes_grayscale

    # low-pass filter for removing blurs and backgrounds
    img1=[]
    if len(img.shape)>2:
        for ii in range(3):
            img1.append( 255 - (ndi.gaussian_filter(255 - img[:,:,ii], 6)) ) # for parasites
            # img1 = 255 - (ndi.gaussian_filter(255 - img1, 3))  # for parasites
        img1=np.asarray(img1)
        img1=np.swapaxes(img1,0,1)
        img1=np.swapaxes(img1,1,2)
    else:
        img1=255 - (ndi.gaussian_filter(255 - img, 6))


    return img1


def edge_detect(imthresh,THRESH1=150, THRESH2=0): ##TODO
    import skimage as sk
    import numpy as np
    import scipy.ndimage as ndi
    from needed_functions import comparetunes, show_images_labels, comparetunes_grayscale
    # comparetunes(img1, [lambda inn, x: 255-inn*((inn<100).astype(np.int))], [['tune']], [[25,50,75,100,125,150,175,190]])
    ###> edge detection:
    #### stage1:
    def createSobelKernel(n):
        side = n * 2 + 3
        Kx = np.zeros([side, side])
        Ky = np.copy(Kx)
        halfside = side // 2
        for i in range(side):
            k = halfside + i if i <= halfside else side + halfside - i - 1
            for j in range(side):
                if j < halfside:
                    Kx[i, j] = Ky[j, i] = j - k
                elif j > halfside:
                    Kx[i, j] = Ky[j, i] = k - (side - j - 1)
                else:
                    Kx[i, j] = Ky[j, i] = 0
        return Kx, Ky
    gradx_kernel, grady_kernel= createSobelKernel(THRESH2) ##(TODO): tune
    gradx=ndi.convolve(imthresh,gradx_kernel)
    grady=ndi.convolve(imthresh,grady_kernel)
    img5=np.hypot(gradx,grady) # sobel gradient intensity
    #### stage2:threholding image for highest values:
    img5=img5/np.max([1.0,np.max(img5)])*255.0
    img6=img5>THRESH1
    #####
    ### normally threshold the edge-detected:
    img6=(img6>0)#.astype(np.int8)
    return img6


def get_connected_components(im):
    import skimage as sk
    ###> now list all cropped chromosomes' image and corresponding mask
    labels=sk.measure.label(im)
    regions=sk.measure.regionprops(labels)
    item_masks=[region.filled_image for region in regions]
    item_crops=[im[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] for region in regions]
    item_images=[crop_image(tup[0],tup[1]) for tup in zip(item_crops,item_masks)]
    ###> prune unwanted ones:
    wantedsizes=np.asarray([np.sum(np.sum(x)) for x in item_masks])>300
    wantedsizes=wantedsizes & (np.asarray([np.sum(np.sum(x))/np.prod(x.shape) for x in item_masks])>0.3)
    item_masks=[item_masks[itm] for itm in np.where(wantedsizes)[0]]
    item_crops=[item_crops[itm] for itm in np.where(wantedsizes)[0]]
    item_images=[item_images[itm] for itm in np.where(wantedsizes)[0]]
    from needed_functions import show_images_labels
    return item_masks, item_crops#show_images_labels((item_images))




def threshold_image(img):
    import numpy as np
    from needed_functions import comparetunes, show_images_labels, comparetunes_grayscale
    isnotint=np.all((img<=1) & (img>=0))
    if isnotint:
        img1=np.floor(255*img)
    img2=img*((img<100).astype(np.int)) ##(TODO): tune thresh
    # comparetunes(img2, [lambda inn, x: 255-inn*((inn<100).astype(np.int))], [['tune']], [[25,50,75,100,125,150,175,190]])
    return img2




def crop_image(image, crop_mask):  #### only for 2d monochrom
    ''' Crop the non_zeros pixels of an image  to a new image
    '''
    from skimage.util import crop
    pxlst = np.where(crop_mask.ravel())[0]
    dims = crop_mask.shape
    imgwidthy = dims[1]  # dimension in y, but in plot being x
    imgwidthx = dims[0]  # dimension in x, but in plot being y
    # x and y are flipped???
    # matrix notation!!!
    pixely = pxlst % imgwidthy
    pixelx = pxlst // imgwidthy
    minpixelx = np.min(pixelx)
    minpixely = np.min(pixely)
    maxpixelx = np.max(pixelx)
    maxpixely = np.max(pixely)
    crops = crop_mask * image  # (TODO): Computational Burden issue
    img_crop = crop(crops, ((minpixelx, imgwidthx - maxpixelx - 1),
                            (minpixely, imgwidthy - maxpixely - 1)))
    return img_crop

def crop_colored_image(imag, crop_mask):  #### only for 2d monochrom
    import numpy as np
    image = np.copy(imag)
    rgbIn1stDimension=image.shape[0]==3
    if rgbIn1stDimension:
        image=image.transpose([1,2,0])
    ''' Crop the non_zeros pixels of an image  to a new image
    '''
    from skimage.util import crop
    pxlst = np.where(crop_mask.ravel())[0]
    dims = crop_mask.shape
    imgwidthy = dims[1]  # dimension in y, but in plot being x
    imgwidthx = dims[0]  # dimension in x, but in plot being y
    # x and y are flipped???
    # matrix notation!!!
    pixely = pxlst % imgwidthy
    pixelx = pxlst // imgwidthy
    minpixelx = np.min(pixelx)
    minpixely = np.min(pixely)
    maxpixelx = np.max(pixelx)
    maxpixely = np.max(pixely)
    crops1 = crop_mask * image[0, :, :] + (255-255*crop_mask)  # (TODO): Computational Burden issue
    crops2 = crop_mask * image[1, :, :] + (255 - 255 * crop_mask)  # (TODO): Computational Burden issue
    crops3 = crop_mask * image[2, :, :] + (255 - 255 * crop_mask)  # (TODO): Computational Burden issue
    img_crop1 = crop(crops1, ((minpixelx, imgwidthx - maxpixelx - 1),
                            (minpixely, imgwidthy - maxpixely - 1)))
    img_crop2 = crop(crops2, ((minpixelx, imgwidthx - maxpixelx - 1),
                            (minpixely, imgwidthy - maxpixely - 1)))
    img_crop3 = crop(crops3, ((minpixelx, imgwidthx - maxpixelx - 1),
                            (minpixely, imgwidthy - maxpixely - 1)))
    img_crop = np.asarray(img_crop1,img_crop2,img_crop3)
    if not rgbIn1stDimension:
        img_crop= np.transpose(img_crop,[1,2,0])
    return img_crop


def detect_crop_components_remove_nonparasites(image=None):
    ######### Input: gets the crops concerning rectangle of interest
    ## using a metric, prune the unwanted connected components of parasites
    from skimage.io import imread, imsave,imshow
    import numpy as np
    import pickle
    if image is None:
        impath='20190525_132818.jpg'
        image= imread(impath)
    ##TODO Enable passing to standard scale: # preprocessor= preprocessing(allimagespath); preprocessor.fit() ; preprocessor.transform()
    image2= denoise_image(image)

    # pass to trhesholding
    image3= threshold_image(image2)

    # eligible now to convert from rgb to gray
    image4 = np.dot(image3, [0.2989, 0.5870, 0.1140])

    # pass to edge detection for later
    image_edge= edge_detect(image4,100,0)
    from needed_functions import comparetunes, show_images_labels, comparetunes_grayscale
    # comparetunes_grayscale(image4, [lambda inn, x: edge_detect(inn,x,0)], [['tune']], [[100,115,130,145,160,175,190]]) ## 100, 115
    # comparetunes_grayscale(image4, [lambda inn, x: edge_detect(inn,100,x)], [['tune']], [[0,1,2,3]]) ## 100, 115

    # pass to connected component finder
    # print('here10')
    masks, crops = get_connected_components(image4)






def detect_crop_components_remove_nonmacrophages(image=None):
    ## using sklearn_lineaar classifier over sift features, prune the unwanted connected components of macrophages over database with labels
    from skimage.io import imread, imsave,imshow
    if image is None:
        impath='C:\\Users\\m3.aaaa\\Desktop\\jahad\\projects2\\leishmania__detection\\Leishmania\\GoldStandardsDrBahreyni\\20190525_132818.jpg'
        image= imread(impath)



def createSobelKernel(n):
    side=n*2+3
    Kx=np.zeros([side,side])
    Ky=np.copy(Kx)
    halfside=side//2
    for i in range(side):
        k= halfside+i if i<=halfside else side+halfside-i-1
        for j in range(side):
            if j<halfside:
                Kx[i,j]=Ky[j,i]=j-k
            elif j>halfside:
                Kx[i,j]=Ky[j,i]=k-(side-j-1)
            else:
                Kx[i,j]=Ky[j,i]=0
    return Kx,Ky

