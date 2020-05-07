###################################################################################################
# Classification of objects using a modified Alex-Net cascaded with reliability metrics
# By Mojtaba Moattari
# To reduce the errors in decision, the outputs have been fed into a metric learning reliability test function, rejecting less certain decisions.
# For more information, I am pleased to be emailed: moatary@aut.ac.ir
###################################################################################################
#global libraries to import:

# import argparse
import os
import time
import shutil
import skimage as sk
from skimage import io, measure
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import scipy.misc
import pickle
from kario_preproc2 import *
from needy_funcs import * #loadTrainLabel #loadTestBatch
from segment_pref1 import *
import torch as t
import torchvision.models as models
from kario_proc3 import ordinary_model
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import PIL.Image as Image


#import gc
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# import sys.path as path
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

# functions needed to define:
def transpose_if_w_over_hfunc(pil):
    return pil.transpose(Image.ROTATE_90) if pil.size[0]>pil.size[1] else pil
def normalze(x):
    return x/np.max(x)
def histeqq(x):
    return sk.exposure.equalize_hist(x)
def np2pil(x):
    return Image.fromarray(x)
def pil2np(x):
    return np.array(x)



def loadargs():
    args=arg()
    args.cudaenabled = False  # false in this model
    # general parameters
    args.batchsize =1 # 
    args.learningrate = 1  # todo:TUNE
    args.momentum = 0.6  # todo:TUNE
    args.weight_decay = 0.5  # todo:TUNE
    args.batchnorm = False  # todo: not implemented yet
    args.batchsampler = False  # weighted #adaptive #todo:not debugged yet :  shuffle/attention / nearestneighbor / ada (adaptive reweighting(increase/decrease) sampler prob based on validation err)
    args.earlystopimprovethresh = False  # todo:not implemented yet
    args.maxiter = 100  #
    args.dim = 10
    args.current_dim = args.dim
    args.ncomp = 1
    args.samplesegment = 0.3  # 0.3
    args.histfuncase = 'gaussian'
    args.hist_cnt = 35#int(max(35, args.data.shape[1] / 100))
    args.gamma_normlossweight = 0.6#0.2
    #
    args.thresh2stop=0.000000001 ##TODO
    args.fuzzymembershiptype = 'gauss'
    args.ggaussbeta = 3
    #
    args.numofchanges_inlradjust=50
    args.lr_reduceweight=0.75
    #
    args.ncomp=10

    #
    return args


class arg:
    pass


class tuner():
    def __init__(self,*tunes):
        names=[]
        self.args=loadargs()
        # get names+  for each tune, index name of key:
        tunes_index=[[] for _ in tunes]
        for ii,tune in enumerate(tunes):
            keys=list(tune.keys())
            for jj,key in enumerate(keys):
                if key not in names:
                    names.append(key)
                tunes_index[ii].append(names.index(key))
        tuneslist=[[[]] for _ in tunes]
        keyslist=[[[]]  for _ in tunes]
        newtuneslist=[[] for _ in tunes]
        newkeyslist=[[]  for _ in tunes]
        currentcnt=1
        # get list:
        for i,(tune,tuneindex) in enumerate(zip(tunes,tunes_index)):
            remaineddict=tune.copy()
            for indofnameind, nameind in enumerate( tuneindex):  # key,values in zip(remaineddict.keys(), remaineddict.values()):
                key = names[nameind]
                values = tune[key]
                for value in values:
                    for keyy,itm in zip(keyslist[i],tuneslist[i]):
                        newtuneslist[i].append([*itm, value ])
                        newkeyslist[i].append([*keyy, key ])
                tuneslist[i]= newtuneslist[i].copy()
                newtuneslist[i]=[]
                keyslist[i]= newkeyslist[i].copy()
                newkeyslist[i]=[]

        # concat results :
        tuness = []
        for itm in tuneslist:
            tuness.extend(itm)
        keyss=[]
        for itm in keyslist:
            keyss.extend(itm)

        # return list of dicts:
        self.listoftunes=[dict(zip(itmkeys,itmvalues)) for itmkeys,itmvalues in zip(keyss,tuness)]
        self.currentindex=-1



    def __iter__(self):
        for self.currentindex in range(self.currentindex,len(self.listoftunes)):
            self.tune= self.listoftunes[self.currentindex]
            yield self.applytune().__dict__
            # yield self.applytune()


    def applytune(self):
        arggs = self.args
        for itmname,itmvalue in zip(self.tune.keys(),self.tune.values()):
            exec('arggs.'+itmname+'='+itmvalue)
        self.args=arggs
        return self.args



def main(ii=0,tuningparams=None): # load data standard pytorchish way:
    from kario_preproc1 import adjust_learning_rate
    from kario_preproc1 import validate,train
    from PIL import Image
    # Tunable parameters:
    LEARNING_RATE =0.037#
    MOMENTUM = 0.8#9
    WGHT_DECAY = 1e-5
    NUM_WORKERS = 4 # 
    BATCH_SIZE =20 # 80
    NUM_EPOCHS = 75 #50
    START_EPOCH = 0
    MODEL_NAMES = ['alexnet', 'densenet121', 'inceptionv3', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'vgg11', 'vgg16', 'vgg11_bn', 'vgg16_bn']
    MODEL_NAME='alexnetmod1'#MODEL_NAMES[0]
    SHUFFLE = False
    PRETRAINED = False
    DISTRIBUTED = False ##(TODO)
    EVALUATION_DATA_READY=True
    PRINT_FREQ = 100
    SAVE_FREQUENCY= 5 ##new
    NUM_CLASSES = 24 ##new
    #
    # tuningparams={}
    #
    path2pickleValidationRes='../benchmarks/resultedbenchmarkresults.val/' + str(ii)+'/'
    try:
        os.mkdir(path2pickleValidationRes)
    except:
        pass
    histeq= transforms.Lambda(histeqq)#lambda x: sk.exposure.equalize_hist(x))
    # Data loading / Transforming / Model loading :
    path2train= '../datapath/traindata_experimental/'
    traindir=path2train #'torchCmptData/train/'
    transpose_if_w_over_h = transforms.Lambda(transpose_if_w_over_hfunc)# transforms.Lambda(lambda pil : pil.transpose(Image.ROTATE_90) if pil.size[0]>pil.size[1] else pil) #tensorish#  transpose_if_w_over_h= transforms.Lambda(lambda tens : tens.transpose(1,2) if tens.shape[2]>tens.shape[1])
    normalize = transforms.Lambda(normalze)#lambda x:x/np.max(x)) #normalize=transforms.Normalize(mean=[0.0],std=[255.0])#normalize = transforms.Normalize(mean=[75.6478, 75.6478, 75.6478],std=[27.4981, 27.4981, 27.4981])
    numpy2pil, pil2numpy =transforms.Lambda(np2pil), transforms.Lambda(pil2np) # warzizg: for coloed iwages it gets trazsposed
    # numpy2pil, pil2numpy =transforms.Lambda(lambda x:Image.fromarray(x)), transforms.Lambda(lambda x:np.array(x)) # warzizg: for coloed iwages it gets trazsposed
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([  ##(TODO):Is it transform pipeline? what are the funcs
            #transforms.RandomResizedCrop(224),
            transforms.Grayscale(),
            pil2numpy,
            histeq,
            normalize,
            numpy2pil,
            transpose_if_w_over_h, #(TODO): Debug
            transforms.Resize((100,200)), # (100,100)
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))  ##(TODO):Is it transform pipeline? what are the funcs
    if DISTRIBUTED:
        train_sampler = t.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = t.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle= True ,num_workers=NUM_WORKERS, pin_memory=True, sampler=train_sampler)
    #####debug::::: aa=iter(train_loader); aaa=next(aa); np.max(aaa[0].numpy())
    #####debug::::: b=list(zip(*list(iter(train_loader)))); s1=[ b[0][i].shape[2] for i in range(len(b[0]))] ; s2=[ b[0][i].shape[3] for i in range(len(b[0]))] ; s1 ; s2
    model, parameters = ordinary_model(MODEL_NAME,NUM_CLASSES,tuningparams=tuningparams) # model = models.__dict__[MODEL_NAME](pretrained=PRETRAINED)
    # classifier criterion loading:
    criterion = t.nn.CrossEntropyLoss()
    # optimizer loading:
    optimizer = t.optim.SGD(parameters,lr=LEARNING_RATE,momentum=MOMENTUM,weight_decay=WGHT_DECAY)
    # optimizer = t.optim.Adam(parameters, lr = 0.0001)(parameters,lr=LEARNING_RATE,momentum=MOMENTUM,weight_decay=WGHT_DECAY)
    # VALIDATE DATA : 4 ONLINE HYPERPARAMETER TUNING
    path2validation = '../datapath/validationdata_experimental/'
    validationdir=path2validation
    val_loader = t.utils.data.DataLoader(
        datasets.ImageFolder(validationdir, transforms.Compose([
            transforms.Grayscale(),
            pil2numpy,
            histeq,
            normalize,
            numpy2pil,
            transpose_if_w_over_h,
            transforms.Resize((100,200)), # (100,100)
            transforms.ToTensor(),
        ])),
        batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, pin_memory = True)

    best_prec1=0
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        if DISTRIBUTED:
            train_sampler.set_epoch(epoch)
        LEARNING_RATE = adjust_learning_rate(optimizer, LEARNING_RATE, epoch)

        # train for one epoch
        print('starting epoch#%i'%(epoch))
        train(train_loader, model, criterion, optimizer, epoch, print_freq = PRINT_FREQ)
        print('endof epoch#%i'%(epoch))
        t.save(model, path2pickleValidationRes+"picklesavecheckpoint%d.trch"%(epoch))

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion,saveDir=path2pickleValidationRes)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if epoch%SAVE_FREQUENCY==SAVE_FREQUENCY-1:

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': MODEL_NAME,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                'tuningparam': tuningparams,
                'tuningii': ii,
            }, is_best,     path2pickleValidationRes,epoch)


    print('')



def save_checkpoint(state, is_best, filename,number):
    full_filename = filename + '/checkpoint.%d.tar'%(number)
    t.save(state, full_filename)
    if is_best:
        new_filename = filename + '/checkpointbest.tar'
        shutil.copyfile(full_filename, new_filename)



def train(train_loader, model, criterion, optimizer, epoch, print_freq = 20):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # err = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # ##(TODO):cuda # target = target.cuda(async=True) ##(MENTION ISCUDA IN INPUT_FUNC)
        input_var = t.autograd.Variable(input)
        target_var = t.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # err.update()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def adjust_learning_rate(optimizer, lr, epoch):
    lr = lr * 0.92**(epoch // 10)
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if epoch == 30 or epoch == 33 or epoch == 36: ##todo: not verified by best acc
    #     lr = lr * 0.8##todo: not verified by best acc
    # # elif epoch >= 50:# or epoch == 33 or epoch == 36:
    # #     lr = lr * (0.8 ** (epoch // 4))
    # else:
    #     lr = lr * (0.7 ** (epoch // 30))  # lr = lr * (0.65 ** (epoch // 36)) # TODO: lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)): ##(TODO):not debugged
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res






def validate(val_loader, model, criterion, saveDir='/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/', print_freq=20):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    file = open( saveDir+'/test.txt',"a")
    for i, (input, target) in enumerate(val_loader):
        #i=0
        # (input, target) = next(iter(val_loader))
        # ##(TODO):cuda #target = target.cuda(async=True)
        input_var = t.autograd.Variable(input, volatile=True)
        target_var = t.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # compute output
        # output = model(input)
        # loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    file.write('{top1.avg:.3f} \n'
          .format(top1=top1))
    file.close()
    return top1.avg


def associateId2DifferentNames(listofimages):
    nums=[int(itm.split('/')[-1].split('_')[0]) for itm in listofimages]
    nums=np.asarray(nums)
    numsinds=np.argsort(nums)
    listofimages=np.asarray(listofimages)[numsinds]
    names = [itm.split('/')[-1].split('_')[-1].split('.')[0] for itm in listofimages]
    diff= [names[i]==names[i+1] for i in range(len(names)-1)]
    diff=[False,*diff]
    return np.cumsum(1-(np.asarray(diff).astype(int))).tolist(), listofimages




def main5_getLabelsFromColors():
    from kario_proc3 import ImageManuallyEditted, getManualTrainLabelsInfo
    from needy_funcs import comparetunes
    import pickle
    import os
    import numpy as np
    import shutil
    #1) get all data from saved main4_DebugDataCleaner1
    with open('/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/pickle1_resultofdatacleaning.pkl', 'rb') as fff:
        pickledata  = pickle.load(fff)
    #2) import colors-labels right data
    datasrc='/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/1stcorrectionphase_path2correctocrs/'
    def getColorIndsLblsVecs(datasrc):
        ColorLbls=np.asarray([int(itm.split(' ')[0].split('_')[-1]) for itm in os.listdir(datasrc) if itm.split('.')[-1]=='png'])
        ColorInds=np.asarray([int(itm.split(' ')[1].split('_')[-1]) for itm in os.listdir(datasrc) if itm.split('.')[-1]=='png'])
        ColorVecs=np.asarray([[int(itm.split('_')[-1].split('.')[0]),int(itm.split('_')[-1].split('.')[1]),int(itm.split('_')[-1].split('.')[2])] for itm in os.listdir(datasrc) if itm.split('.')[-1] == 'png'])
        ck = [len(itm) != 3 for itm in ColorVecs]
        ColorVecs = ColorVecs[np.where(np.asarray(ck) == False)[0].tolist()]
        ColorVecs = np.asarray(ColorVecs.tolist())
        #4) get labels from colors:
        current_existing_labels=np.sort(np.unique(ColorLbls))
        current_existing_colors=np.concatenate([np.reshape(np.mean(ColorVecs[np.where(ColorLbls==lbl)[0].tolist()],0),[1,3]) for lbl in current_existing_labels],axis=0)
        return ColorLbls,ColorInds,ColorVecs,current_existing_labels,current_existing_colors
    # # get current color info:
    # ColorLbls, ColorInds, ColorVecs, current_existing_labels, current_existing_colors = getColorIndsLblsVecs(datasrc)
    # #5)change label x to 24:
    # ColorLbls1, ColorInds1, ColorVecs1, current_existing_labels1, current_existing_colors1 = getColorIndsLblsVecs(datasrc+'label_x')
    # ColorLbls1[ColorLbls1==22]=24
    # current_existing_labels1[current_existing_labels1==22]=24
    # ColorLbls, ColorInds, ColorVecs, current_existing_labels, current_existing_colors = np.append(ColorLbls,ColorLbls1), np.append(ColorInds,ColorInds1), np.append(ColorVecs,ColorVecs1), np.append(current_existing_labels,current_existing_labels1), np.concatenate([current_existing_colors,current_existing_colors1],0),
    # ##(TODO): REVISED LABEL OF INDS HAS TO BE SAVED AND REPLACED WITH CURRENT PICKLE-BACKUP
    # # ANd the resulted color-label-map :
    # current_existing_colors=np.asmatrix(current_existing_colors)/np.transpose([np.sum(current_existing_colors**2,1)**0.5],[1,0])
    #
    #
    # # Load all data needed to label to find label using the created color-label-map:
    # WrongLbls, WrongInds,WrongVecs ,tmp1 ,tmp1  = getColorIndsLblsVecs(datasrc+'wronglabels_mustgetright')
    # getColorIndx=lambda color:  current_existing_labels[ np.argmax(    current_existing_colors*(np.asarray(color).astype(np.float64).reshape([-1,1]))    ) ]
    # RightLbls= [getColorIndx(itm) for itm in WrongVecs ]
    # # now, save new names to recheck:
    # where2save=datasrc + 'wronglabels_mustberight/'
    # try:
    #     os.mkdir(where2save)
    # except:
    #     pass
    # for i in range(len(WrongInds)):
    #     srcfile= datasrc + 'wronglabels_mustgetright/' + 'label_%02i number_%02i color_%03i.%03i.%03i.png' % (WrongLbls[i], WrongInds[i], WrongVecs[i][0], WrongVecs[i][1],WrongVecs[i][2])
    #     dstfile= where2save + 'label_%02i number_%02i color_%03i.%03i.%03i.png' % (RightLbls[i], WrongInds[i], WrongVecs[i][0], WrongVecs[i][1],WrongVecs[i][2])
    #     shutil.copyfile(srcfile,dstfile)
    #     os.chmod(dstfile,0o666 )
    ## get color-label map from list of corrected images (last time done)
    rightlabelspath='/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/truelabelswithout5,15,16/'
    ColorLbls, ColorInds, ColorVecs, current_existing_labels, current_existing_colors = getColorIndsLblsVecs(rightlabelspath)
    # look for labels 5,15,16:
    colorlabelmap_txt='/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/colorlabelmap.txt'
    # # update indice-labels in pickle,trainfolder    ///  update color-label-map and save as pickle
    # truelabelswithout5_15_16='/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/truelabelswithout5,15,16/'
    # ColorLbls, ColorInds, ColorVecs, current_existing_labels, current_existing_colors = getColorIndsLblsVecs(truelabelswithout5_15_16)
    # with open(colorlabelmap_txt,'wb') as fil:
    #     pickle.dump([current_existing_labels, current_existing_colors],fil)
    # ## load indice-labels-map from file:
    with open(colorlabelmap_txt,'rb') as fil:
        current_existing_labels, current_existing_colors=pickle.load(fil)
    # current_existing_colors=np.asmatrix(current_existing_colors/np.transpose([(np.sum(current_existing_colors**2,1))**0.5],[1,0]))
    # map for dataset of mixed right wrong labels
    WrongLbls, WrongInds,WrongVecs, tmp1, tmp1  = getColorIndsLblsVecs(datasrc)
    # getColorIndx=lambda color:  current_existing_labels[ np.argmax(    current_existing_colors*(np.asarray(color).astype(np.float64).reshape([-1,1]))    ) ]
    getColorIndx=lambda color:  current_existing_labels[ np.argmax(    current_existing_colors*(np.asarray(color).astype(np.float64).reshape([-1,1]))    ) ]
    getcolormult=lambda color:   np.max(    current_existing_colors*(np.asarray(color).astype(np.float64).reshape([-1,1]))    )
    def getColorIndx(color):
        thresh2reject= 0.3
        color1=color/(np.sum(color**2))**0.5
    RightLbls= [getColorIndx(itm) for itm in WrongVecs ]
    RightMult= [getcolormult(itm/(np.sum(itm**2))**0.5) for itm in WrongVecs ]
    # save them to wronglabels_mustberight:
    where2save=datasrc+'wronglabels_mustberight/'
    try:
        os.mkdir(where2save)
    except:
        pass
    for i in range(len(WrongInds)):
        srcfile= datasrc +  'label_%02i number_%02i color_%03i.%03i.%03i.png' % (WrongLbls[i], WrongInds[i], WrongVecs[i][0], WrongVecs[i][1],WrongVecs[i][2])
        dstfile= where2save + 'label_%02i number_%02i color_%03i.%03i.%03i.png' % (RightLbls[i], WrongInds[i], WrongVecs[i][0], WrongVecs[i][1],WrongVecs[i][2])
        shutil.copyfile(srcfile,dstfile)
        os.chmod(dstfile,0o666 )
    ids=[9914,10198,79,9646,10979,7661,5730,10238,8119,10508,8465,1034,1837,10060,8195,160,6890,6724,3640,2143,2565,794,1986,71,1364,11057,3361,4327]
    def showsomeimages(ids):
        pat='/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/1stcorrectionphase_path2correctocrs/wronglabels_mustberight/'
        imgs=os.listdir(pat)
        inds=[int(itm.split(' ')[1].split('_')[1]) for itm in imgs if itm.split('.')[-1]=='png']
        ims=[]
        for eachid in ids:
            w=inds.index(eachid)
            ims.append(sk.io.imread(pat+imgs[w]))
        show(*ims,labels=ids)
    ## indices with totally right labels except 5,15,16:
    ColorLbls, ColorInds, tmp0, tmp1, tmp2 = getColorIndsLblsVecs(where2save)
    ColorLbls, ColorInds =ColorLbls.tolist(), ColorInds.tolist()
    path2folders=where2save+'mustgetright/'
    dirs=os.listdir(path2folders)
    #froms=[int(itm.split(' ')[0]) for itm in dirs]
    tos=  [int(itm.split(' ')[1]) for itm in dirs]
    for fldnum in range(len(dirs)):
        tmp3, ColorInds1, tmp0, tmp1, tmp2 = getColorIndsLblsVecs(path2folders+dirs[fldnum])
        ColorLbls1= [tos[fldnum]]*len(ColorInds1)
        ColorInds.append(ColorInds1)
        ColorLbls.append(ColorLbls1)
    # save the indx-labels by pickle:
    with open(datasrc+'indxlabelsWO5,15,16.pkl','wb') as ffff:
        pickle.dump([ColorInds,ColorLbls],ffff)
    TrainVecColors=np.asarray(TrainVecs)
    #setdiff=lambda inn,out: [x for x in inn if x not in out]
    #unexistedLbls=np.asarray(setdiff(np.asarray(list(range(1,23))),current_existing_labels)) # 1 2 3 4 5 8 15 16,[163,80,244]
    #unexistedColors=np.asarray([[190,141,141], [117,180,117] , [138,138,188], [192,144,192], [194,194,194] , [235,185,185], [162,244,78]])
    current_existing_labels=np.append(current_existing_labels,unexistedLbls)
    current_existing_colors=np.append(current_existing_colors,unexistedColors.astype(np.float64),0)
    current_existing_colors=np.asarray(current_existing_colors)/(np.sqrt(np.asarray(np.repeat(np.sum(np.asarray(current_existing_colors)**2,1).reshape([1,-1]) ,  current_existing_colors.shape[1],0  )))).T
    current_existing_colors=np.asmatrix(current_existing_colors )
    ## NOW, extract each labels using nearest neighbor of colors
    TrainLbls = [getColorIndx(itm) for itm in TrainVecs]

def main6_findChromosomePanj():  # MAKING PICKLES AND TRAINLABEL DATA (BUT DIRTY AND MUST BE PRUNED)
    from kario_proc3 import ImageManuallyEditted, getManualTrainLabelsInfo
    from needy_funcs import comparetunes
    import pickle
    import numpy as np
    import os
    import shutil
    import skimage as sk
    from segment_pref1 import findchromosome5boundary
    #1) loading images with indices
    with open('/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/pickle1_resultofdatacleaning.pkl', 'rb') as fff:
        preproc_indices, preproc_dataindices, preproc_chromeindices, preproc_subjectid, preproc_labels, preproc_colors, preproc_scores, preproc_images, preproc_innerboundarys3pcs, preproc_innerboundarys3pc_lastend  = pickle.load(fff)

    #2) loading list of images those indices belong to
    path2pickle_listofdataimage='/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/pickle_listofdataimage.pkl'
    with open(path2pickle_listofdataimage, 'rb') as file:
        listofimages, id4eachimage= pickle.load(file)
    verified_dataindices=np.unique(preproc_dataindices)
    path2data = '/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/datacleaning/'
    path2pickle, path2picklebackup = '/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/pickle1_resultofdatacleaning_chromosome5.pkl', '/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/picklebackup1_resultofdatacleaning_chromosome5.pkl'
    path2correctocrs_1stcorrectionphase = '/home/m2/Desktop/jahad_l/Karyotype_97/_converteed2tif/part4coding/1stcorrectionphase_path2correctocrs_chromosome5/'
    if not os.path.isdir(path2correctocrs_1stcorrectionphase):
        os.mkdir(path2correctocrs_1stcorrectionphase)
    id4eachimage=[preproc_subjectid[itm] for itm in verified_dataindices]
    listofimages=[listofimages[itm] for itm in verified_dataindices]
    all_res = []
    wrongocrs = []
    currentindex, preproc_dataindices, preproc_chromeindices = - 1, [], []
    preproc_indices, preproc_subjectid, preproc_labels, preproc_scores, preproc_colors, preproc_images, preproc_innerboundarys3pcs,preproc_innerboundarys3pc_lastend = [], [], [], [], [], [], [],[]
    for i in range(len(listofimages)):  # ,image_name in enumerate():49 * 20 + 1,
        image_name = listofimages[i]
        # if i!=4:
        #     continue
        default_image = sk.io.imread(image_name)
        res = findchromosome5boundary(default_image, plott=False)
        tmp, chromes_scores, chromes_reginds, chromes_innerboundarys3pc_lastend, chromes_innerboundarys3pcs, chromes_imgs, chromes_bbox, chromes_masks, chromes_outerboundarys, chromes_recognizedlabels, chromes_markedimgs, chromes_colors = res
        all_res.append(res)
        # save images to manually find wrong labels:
        for inum, lbl in enumerate(chromes_recognizedlabels):
            currentindex += 1
            preproc_dataindices.append(verified_dataindices[i])
            preproc_subjectid.append(id4eachimage[i])
            preproc_labels.append(chromes_recognizedlabels[inum])
            preproc_colors.append(chromes_colors[inum])
            preproc_scores.append(chromes_scores[inum])
            preproc_images.append(chromes_imgs[inum].tolist())
            preproc_innerboundarys3pcs.append(chromes_innerboundarys3pcs[inum])
            preproc_innerboundarys3pc_lastend.append(chromes_innerboundarys3pc_lastend[inum])
            path2sav = path2correctocrs_1stcorrectionphase + 'label_%02i number_%02i color_%03i.%03i.%03i.png' % (
            chromes_recognizedlabels[inum], currentindex, chromes_colors[inum][0], chromes_colors[inum][1],
            chromes_colors[inum][2])
            scipy.misc.imsave(path2sav, chromes_markedimgs[inum])
        if i % 20 == 0:
            print('checkpoint%00i' % (i // 20))
            try:
                shutil.copyfile(path2pickle, path2picklebackup)
            except:
                pass
            with open(path2pickle, "wb") as picklefile:
                pickledata = [preproc_indices, preproc_dataindices, preproc_chromeindices, preproc_subjectid, preproc_labels, preproc_colors, preproc_scores, preproc_images, preproc_innerboundarys3pcs,preproc_innerboundarys3pc_lastend]
                pickle.dump(pickledata, picklefile)
    with open(path2pickle, "wb") as picklefile:
        pickledata = [preproc_indices, preproc_dataindices, preproc_chromeindices, preproc_subjectid, preproc_labels, preproc_colors, preproc_scores, preproc_images, preproc_innerboundarys3pcs,preproc_innerboundarys3pc_lastend]
        pickle.dump(pickledata, picklefile)



















if __name__ == '__main__':
    tunes = {'nkernels': ['19','15','12','8','4'], 'nclassif': ['18', '14', '10', '7','4']}  ##new
    tunerobj = tuner(tunes)
    #
    main3(1, {'nkernels': 15, 'nclassif': 18})
    for ii, tuningparams in enumerate(tunerobj):
        if ii==1:
            continue
        main(ii, tuningparams)
