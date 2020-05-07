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
### class for automatically check all new parameter settings and finding teh best tunign parameter in all their interactions
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
    ### Adjusting rate of learning for avoiding premature convergence
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
    # keeping track of average
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
    # computing classification accuracy
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
    # validating accuracy
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






