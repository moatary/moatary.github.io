#########################################################################################################################
### Embedding higher-orders of entity adjacencies by a new deep Maximum Dependency Minimum Redundancy framework
# In this project, shared representation between different orders of Knowledge Graph adjacencies is sought 
# using a new Maximum Derpendency Minimum Redundancy (MDMR) framework. There are a lot of redundancies in 
# the knowledge graph entities interrelations, per each order of contingency that should be removed. 
# Deep Independent Component Analysis is a powerful bioinspired deep model, for reducing redundancy in data 
# for finding the minimal number of principal latent dimension. I used Bengio's NICE deep neural net
# for Minimum Redundancy aspect of MDMR. For Maximum inter-adjacency-order Dependency, I used 
# Maximum Entropy and Cross-Correlation functions. The resulted embeddings improved Knowledge Graph Completion 
# in Freebase substantially.
### By Mojtaba Moattari
### I am pleased to elaborate more upon this code and how to use the model. Please reach me by moatary@aut.ac.ir
#######################################################################################################################

import torch as t
import torch.nn as nn
import numpy as np
from utils import xavier_init
from datastructs import Vocab
class model_a (t.nn.Module): #@# model with neural net+nolearnable embeddings+ hiersoftmax
    ## This model is also criterion / outputs a loss to backward
    def __init__(self, args):
        #self.args = args
        """
        Creating   model.
        """
        pass




class model_b (t.nn.Module): #@# model with learnable embedding +clusterbasedParamterershift+ bilinear trans
    def __init__(self, args):
        super(model_b,self).__init__()
        """
        Creating   model.
        """
        self.cudaenabled=args.cudaenabled
        # general parameters
        self.batchsize=args.batchsize # todo: Felan 1 .# todo:TUNE
        self.learningrate =args.learningrate # todo:TUNE
        self.momentum =args.momentum   # todo:TUNE
        self.weight_decay =args.weight_decay  # todo:TUNE
        self.batchnorm =args.batchnorm # todo: not implemented yet
        self.batchsampler =args.batchsampler   # weighted #adaptive #todo:not debugged yet :  shuffle/attention / nearestneighbor / ada (adaptive reweighting(increase/decrease) sampler prob based on validation err)
        self.earlystopimprovethresh =args.earlystopimprovethresh  # todo:not implemented yet
        self.maxiter = args.maxiter   #
        # glossaries to load
        self.datatype= args.datatype #['weighted','unnestedrels','unweightednestsinglecount'] #, 'unweightednest'
        self.entities_embeds=args.entities.todense() #t.from_numpy(args.entities.todense())
        self.relations_embeds =args.relations.todense() #t.from_numpy(args.relations.todense())
        self.edim= self.entities_embeds.shape[0]
        self.rdim= self.relations_embeds.shape[0]
        self.reldim=args.reldim
        self.entdim=args.entdim
        # todo: revise for better memory: tmp=vocab; del vocab
        self.entities_clusters_id= args.entities_clusters_id#
        self.id2clustersid4entities= Vocab(self.entities_clusters_id).word2id#args.entities_clusters_id
        self.relations_clusters_id= args.relations_clusters_id#
        self.id2clustersid4relations= Vocab(self.relations_clusters_id).word2id
        self.N_c_e= len(self.id2clustersid4entities)
        self.N_c_r= len(self.id2clustersid4relations)
        #
        self.tails_glossary_tensor= args.tails_glossary_tensor #todo
        self.negative_sampler_thresh = args.negative_sampler_thresh ##TODO:tune
        self.projElossfcnType= args.projElossfcnType ##TODO:tune
        self.cluster_update_regularity = args.cluster_update_regularity ##TODO:tune
        self.pretrained_entities_dimentionality_reduction=args.pretrained_entities_dimentionality_reduction ##TODO:tune
        self.pretrained_relations_dimentionality_reduction=args.pretrained_relations_dimentionality_reduction ##TODO:tune
        self.normalize_candidate_tails=args.normalize_candidate_tails # True or false ##TODO:tune, reweight all tails DOWN if corresponding head(~input) are dense in numof tails
        self.normalize_candidate_heads=args.normalize_candidate_heads # True or false## ##TODO:tune, reweight each tail DOWN if its connected more densely to more heads
        self.sumoverheads= self.tails_glossary_tensor.sumover('heads')
        self.sumovertails= self.tails_glossary_tensor.sumover('tails')
        self.regularize_within_clusters_relations= args.regularize_within_clusters_relations ##TODO:tune
        self.regularize_within_clusters_entities= args.regularize_within_clusters_entities ##TODO:tune
        self.regularization_rate_relations= args.regularization_rate_relations ##TODO:tune
        self.regularization_rate_entities= args.regularization_rate_entities ##TODO:tune
        # glossaries to learn
        self.E = xavier_init((self.edim,self.entdim)) # entity vecs to indirectly learn
        self.R = xavier_init((self.rdim,self.reldim)) # relation vecs to indirectly learn
        self.CR= xavier_init((self.N_c_r, self.reldim)) # Parameters of all RELATIONS' clusters to indirectly learn
        self.CE= xavier_init((self.N_c_e, self.entdim)) # Parameters of all ENTITIES' clusters to indirectly learn
        # Now set parameters : # t.nn.Parameter sets requires_grad true /also easily lists all gradablevars by model.parameters()
        self.e=self.tnnparameter(t.from_numpy(args.entities[0,:].todense())) #entity_current_input  defaultly,requires_grad=True
        self.r=self.tnnparameter(t.from_numpy(args.relations[0,:].todense())) # relation_current_input # defaultly,requires_grad=True
        self.cr=self.tnnparameter(t.from_numpy(self.CR[0,:])) # entities_clusters # defaultly,requires_grad=True
        self.ce=self.tnnparameter(t.from_numpy(self.CE[0,:])) # relations_clusters # defaultly,requires_grad=True
        # set parameters that don't need update_entities_relations_parameters
        self.bp=self.tnnparameter(t.from_numpy(xavier_init((1, self.entdim)) )) # projection bias # defaultly,requires_grad=True
        # dimensionality reduction, entities' feature  glossaries
        if self.pretrained_entities_dimentionality_reduction is True:
            if args.entities_dimentionality_reduction is not None:
                self.dimred_e = self.fromnumpy(args.entities_dimentionality_reduction)
            else:
                self.dimred_e = self.fromnumpy(xavier_init((args.entities.shape[1], self.edim)))
        else:
            if args.entities_dimentionality_reduction is not None:
                self.dimred_e = self.tnnparameter(t.from_numpy(args.entities_dimentionality_reduction))
            else:
                self.dimred_e = self.tnnparameter(t.from_numpy(xavier_init((args.entities.shape[1], self.edim))))
        # dimensionality reduction, relations' feature  glossaries
        if self.pretrained_relations_dimentionality_reduction is True:
            if args.relations_dimentionality_reduction is not None:
                self.dimred_r = self.fromnumpy(args.relations_dimentionality_reduction)
            else:
                self.dimred_r = self.fromnumpy(xavier_init((args.relations.shape[1], self.rdim)))
        else:
            if args.entities_dimentionality_reduction is not None:
                self.dimred_r = self.tnnparameter(t.from_numpy(args.relations_dimentionality_reduction))
            else:
                self.dimred_r = self.tnnparameter(t.from_numpy(xavier_init((args.relations.shape[1], self.rdim))))
        # current ongoing entity/rel/ParamRegardClus to learn
        self.entity_current_id= 0
        self.relation_current_id= 0
        self.entity_cluster_current_id= 0
        self.relation_cluster_current_id= 0
        # define loss function for model:
        if self.projElossfcnType=='pointwise':
            self.projBfcntype= t.nn.functional.sigmoid
        elif self.projElossfcnType== 'listwise':
            self.projBfcntype= t.nn.functional.softmax


    def fromnumpy(self,dat):
        import torch as t
        if self.cudaenabled==True:
            return t.from_numpy(dat).cuda()
        else:
            return t.from_numpy(dat)


    def tnnparameter(self,tens):
        import torch as t
        if self.cudaenabled==True:
            return t.nn.Parameter(tens.cuda())
        else:
            return t.nn.Parameter(tens)

    def tensors2cuda(self):
        for itm in self.__dict__.keys():
            if type(self.__dict__[itm]) is t.Tensor:
                self.__dict__[itm]= self.__dict__[itm].cuda()

    def setmode(self,mode='train'):
        if mode=='train' or mode=='training':
            self.mode='train'
        elif mode=='test' or mode=='testing' or mode=='valid' or mode=='validation':
            self.mode=='test'


    def forward(self,input_headRelIds):
        from scipy import sparse
        # Only one instance (here 1x2 [relid,entid]) gets passed to the fun
        # input_headEntIds is a tensor of batch of selected head and its relation id
        self.relation_current_id=input_headRelIds[0][0].tolist()
        self.head_current_id=input_headRelIds[0][1].tolist()
        self.relation_current_cluster_id= self.relations_clusters_id[self.relation_current_id]  ##TODO: TEST IT VOCAB RIGHT OR WRONG
        self.head_current_cluster_id= self.entities_clusters_id[self.head_current_id]  ##TODO: TEST IT VOCAB RIGHT OR WRONG
        # get embeds for each:
        self.De = self.fromnumpy(((self.entities_embeds[self.head_current_id, :]))) #self.De= self.fromnumpy((np.asarray(self.entities_embeds[self.head_current_id,:])[0]))
        self.Dr = self.fromnumpy(((self.relations_embeds[self.relation_current_id, :])))# self.Dr= self.fromnumpy((np.asarray(self.relations_embeds[self.relation_current_id,:])[0]))
        # get positive candidates, sample negative candidates: (positively samples all, negatively samples some)
        self.ps=self.tails_glossary_tensor[self.head_current_id, self.relation_current_id, :]
        self.psid= self.tails_glossary_tensor.where[self.head_current_id, self.relation_current_id, :]
        negative_cnt= self.ps.shape[1]-np.sum(self.ps!=0)
        probs4neg= np.random.random(self.ps.shape)
        self.ns= (sparse.lil_matrix(np.ones(self.ps.shape))-self.ps).multiply(probs4neg<self.negative_sampler_thresh).tolil()
        self.nsid= self.ns.nonzero()[1]
        # get params for each:
        self.e=self.tnnparameter( t.from_numpy(self.E[self.head_current_id,:].transpose()).data) # ALSO THIS SUSTAINS OBJ w1.data -= learning_rate * w1.grad.data
        self.r=self.tnnparameter(t.from_numpy(self.R[self.relation_current_id,:].transpose()).data) # ALSO THIS SUSTAINS OBJ w1.data -= learning_rate * w1.grad.data
        self.cr=self.tnnparameter( t.from_numpy(self.CR[ self.relation_current_cluster_id ,:].transpose()).data)
        self.ce=self.tnnparameter( t.from_numpy(self.CE[self.head_current_cluster_id,:].transpose()).data)
        # get (clus regul)params for each:
        if self.regularize_within_clusters_relations == True:
            ids4relationsclus=list(set(self.id2clustersid4relations[self.relation_current_cluster_id])-set([self.relation_current_cluster_id]))
            self.clusrelations= self.tnnparameter( t.from_numpy(self.R[ids4relationsclus,:]) )# except for the main entity
            self.clusrelations_plus_mainrelation = t.cat((self.clusrelations,self.r.unsqueeze(0)),dim=0)
            lossreguls1 = self.regularization_rate_relations * t.sqrt(t.sum(self.clusrelations_plus_mainrelation.var(0)))
        else:
            lossreguls1=t.tensor(0)
        if self.regularize_within_clusters_entities == True:
            ids4entitiesclus=list(set(self.id2clustersid4entities[self.head_current_cluster_id])-set([self.head_current_cluster_id]))
            self.clusentities= self.tnnparameter( t.from_numpy(self.E[ids4entitiesclus,:]) ) # except for the main entity
            self.clusentities_plus_mainentity = t.cat((self.clusentities,self.e.unsqueeze(0)),dim=0)
            lossreguls2= self.regularization_rate_entities * t.sqrt(t.sum(self.clusentities_plus_mainentity.var(0)))
        else:
            lossreguls2 = t.tensor(0)
        lossreguls= lossreguls1+lossreguls2
        # pass first layer of projB:
        lyr1= (self.De.mm( self.dimred_e)*(self.e.unsqueeze(0)) + self.ce.unsqueeze(0)).t() * (self.Dr.mm( self.dimred_r)*(self.r.unsqueeze(0)) + self.cr.unsqueeze(0))
        # pass activation function f : here tanh
        lyr1_act=t.tanh(lyr1)
        # pass next layer of projB:
        lyr2_pos= self.fromnumpy(self.entities_embeds[self.psid,:]).mm( self.dimred_e).mm(lyr1_act.mm(self.r.unsqueeze(0).t())+self.bp.t())
        lyr2_neg= self.fromnumpy(self.entities_embeds[self.nsid,:]).mm( self.dimred_e).mm(lyr1_act.mm(self.r.unsqueeze(0).t())+self.bp.t())
        # pass activation function g: here softmax
        lyr2_pos_act= self.projBfcntype(lyr2_pos,0)
        lyr2_neg_act= self.projBfcntype(lyr2_neg,0)
        # pass to margin-based loss function  # get cross entropy with its corresponding tails as outputs (supervised weighted labels)
        if self.mode == 'train':
            vec_of_relids_rep_neg= [self.relation_current_id for _ in range(len(self.nsid))]
            lossneg= -self.fromnumpy(self.ns[0,self.nsid]/(1+self.sumoverheads[vec_of_relids_rep_neg, self.nsid])/(1+self.sumovertails[self.relation_current_id, self.head_current_id]) ).mm(t.log(1-lyr2_neg_act+1e-9))
            vec_of_relids_rep_pos= [self.relation_current_id for _ in range(len(self.psid))]
            losspos= -self.fromnumpy(self.ps[0,self.psid]/(1+self.sumoverheads[vec_of_relids_rep_pos, self.psid])/(1+self.sumovertails[self.relation_current_id, self.head_current_id]) ).mm(t.log(lyr2_pos_act+1e-9))
            loss=lossneg+losspos+lossreguls
     
            # redefine the optimizer:
            self.opt=t.optim.SGD(self.parameters(),lr=0.05,momentum=0.3,weight_decay=self.weight_decay) ##TODO: Add args' tunable arguments
            self.opt.zero_grad()
            # backward the loss to compute gradients
            loss.backward()
            # update the data :
            self.update_entities_relations_parameters(self.head_current_id,self.relation_current_id,self.head_current_cluster_id,self.relation_current_cluster_id,ids4relationsclus,ids4entitiesclus)

            ##TODO
        elif self.mode=='test':
            vec_of_relids_rep_neg= [self.relation_current_id for _ in range(len(self.nsid))]
            lossneg= -self.fromnumpy(self.ns[0,self.nsid]/(1+self.sumoverheads[vec_of_relids_rep_neg, self.nsid]) ).mm(t.log(1-lyr2_neg_act+1e-9))
            vec_of_relids_rep_pos= [self.relation_current_id for _ in range(len(self.psid))]
            losspos= -self.fromnumpy(self.ps[0,self.psid]/(1+self.sumoverheads[vec_of_relids_rep_pos, self.psid]) ).mm(t.log(lyr2_pos_act+1e-9))
            pass
        if self.cudaenabled==False:
            return [lossneg.data.numpy().squeeze(0).squeeze(0).tolist(),losspos.data.numpy().squeeze(0).squeeze(0).tolist(),lossreguls.data.numpy().squeeze(0).squeeze(0).tolist()] #[loss.data.numpy().squeeze(0).squeeze(0).tolist()]
        else:
            return [lossneg.data.cpu().numpy().squeeze(0).squeeze(0).tolist(),losspos.data.cpu().numpy().squeeze(0).squeeze(0).tolist(),lossreguls.data.cpu().numpy().squeeze(0).squeeze(0).tolist()] #[loss.data.cpu().numpy().squeeze(0).squeeze(0).tolist()]





    def update_entities_relations_parameters(self,entid,relid,entclid,relclid,ids4relationsclus,ids4entitiesclus):
        ##note: self.dimred_e self.dimred_r are automatically retained and learned and don't need this func
        # once grads are available, the parameters get updated:
        self.opt.step()
        # now all the changed data gets resaved back to the glossaries:
        if self.cudaenabled==False:
            self.E[ids4entitiesclus,:]=self.clusentities.data.numpy() # entity vecs to indirectly learn
            self.R[ids4relationsclus,:]=self.clusrelations.data.numpy() # relation vecs to indirectly learn
            self.E[entid,:]=self.e.data.numpy() # entity vecs to indirectly learn
            self.R[relid,:]=self.r.data.numpy() # relation vecs to indirectly learn
            self.CE[entclid,:]=self.ce.data.numpy() # Parameters of all ENTITIES' clusters to indirectly learn
            self.CR[relclid,:]=self.cr.data.numpy() # Parameters of all RELATIONS' clusters to indirectly learn
        else:
            self.E[ids4entitiesclus, :] = self.clusentities.data.cpu().numpy()  # entity vecs to indirectly learn
            self.R[ids4relationsclus, :] = self.clusrelations.data.cpu().numpy()  # relation vecs to indirectly learn
            self.E[entid, :] = self.e.data.cpu().numpy()  # entity vecs to indirectly learn
            self.R[relid, :] = self.r.data.cpu().numpy()  # relation vecs to indirectly learn
            self.CE[entclid, :] = self.ce.data.cpu().numpy()  # Parameters of all ENTITIES' clusters to indirectly learn
            self.CR[relclid, :] = self.cr.data.cpu().numpy()  # Parameters of all RELATIONS' clusters to indirectly learn







class model_b_alldataingpu(t.nn.Module): #@# model with learnable embedding +clusterbasedParamterershift+ bilinear trans
    def __init__(self, args):
        super(model_b_alldataingpu,self).__init__()
        """
        Creating   model.
        """
        self.cudaenabled=args.cudaenabled
        # general parameters
        self.batchsize=args.batchsize # todo: Felan 1 .# todo:TUNE
        self.learningrate =args.learningrate # todo:TUNE
        self.momentum =args.momentum   # todo:TUNE
        self.weight_decay =args.weight_decay  # todo:TUNE
        self.batchnorm =args.batchnorm # todo: not implemented yet
        self.batchsampler =args.batchsampler   # weighted #adaptive #todo:not debugged yet :  shuffle/attention / nearestneighbor / ada (adaptive reweighting(increase/decrease) sampler prob based on validation err)
        self.earlystopimprovethresh =args.earlystopimprovethresh  # todo:not implemented yet
        self.maxiter = args.maxiter   #
        # glossaries to load
        self.datatype= args.datatype #['weighted','unnestedrels','unweightednestsinglecount'] #, 'unweightednest'
        self.entities_embeds=self.fromnumpy(args.entities.todense()) #t.from_numpy(args.entities.todense())
        self.relations_embeds =self.fromnumpy(args.relations.todense() )#t.from_numpy(args.relations.todense())
        self.edim= self.entities_embeds.shape[0]
        self.rdim= self.relations_embeds.shape[0]
        self.reldim=args.reldim
        self.entdim=args.entdim
        # todo: revise for better memory: tmp=vocab; del vocab
        self.entities_clusters_id= args.entities_clusters_id#
        self.id2clustersid4entities= Vocab(self.entities_clusters_id).word2id#args.entities_clusters_id
        self.relations_clusters_id= args.relations_clusters_id#
        self.id2clustersid4relations= Vocab(self.relations_clusters_id).word2id
        self.N_c_e= len(self.id2clustersid4entities)
        self.N_c_r= len(self.id2clustersid4relations)
        #
        self.tails_glossary_tensor= args.tails_glossary_tensor #todo
        self.negative_sampler_thresh = args.negative_sampler_thresh ##TODO:tune
        self.projElossfcnType= args.projElossfcnType ##TODO:tune
        self.cluster_update_regularity = args.cluster_update_regularity ##TODO:tune
        self.pretrained_entities_dimentionality_reduction=args.pretrained_entities_dimentionality_reduction ##TODO:tune
        self.pretrained_relations_dimentionality_reduction=args.pretrained_relations_dimentionality_reduction ##TODO:tune
        self.normalize_candidate_tails=args.normalize_candidate_tails # True or false ##TODO:tune, reweight all tails DOWN if corresponding head(~input) are dense in numof tails
        self.normalize_candidate_heads=args.normalize_candidate_heads # True or false## ##TODO:tune, reweight each tail DOWN if its connected more densely to more heads
        self.sumoverheads= self.tails_glossary_tensor.sumover('heads')
        self.sumovertails= self.tails_glossary_tensor.sumover('tails')
        self.regularize_within_clusters_relations= args.regularize_within_clusters_relations ##TODO:tune
        self.regularize_within_clusters_entities= args.regularize_within_clusters_entities ##TODO:tune
        self.regularization_rate_relations= args.regularization_rate_relations ##TODO:tune
        self.regularization_rate_entities= args.regularization_rate_entities ##TODO:tune
        # glossaries to learn
        self.E = self.fromnumpy(xavier_init((self.edim,self.entdim))) # entity vecs to indirectly learn
        self.R = self.fromnumpy(xavier_init((self.rdim,self.reldim))) # relation vecs to indirectly learn
        self.CR= self.fromnumpy(xavier_init((self.N_c_r, self.reldim))) # Parameters of all RELATIONS' clusters to indirectly learn
        self.CE= self.fromnumpy(xavier_init((self.N_c_e, self.entdim))) # Parameters of all ENTITIES' clusters to indirectly learn
        # Now set parameters : # t.nn.Parameter sets requires_grad true /also easily lists all gradablevars by model.parameters()
        self.e=self.tnnparameter(t.from_numpy(args.entities[0,:].todense())) #entity_current_input  defaultly,requires_grad=True
        self.r=self.tnnparameter(t.from_numpy(args.relations[0,:].todense())) # relation_current_input # defaultly,requires_grad=True
        self.cr=self.tnnparameter(self.CR[0,:].clone()) # entities_clusters # defaultly,requires_grad=True
        self.ce=self.tnnparameter(self.CE[0,:].clone()) # relations_clusters # defaultly,requires_grad=True
        # set parameters that don't need update_entities_relations_parameters
        self.bp=self.tnnparameter(t.from_numpy(xavier_init((1, self.entdim)) )) # projection bias # defaultly,requires_grad=True
        # dimensionality reduction, entities' feature  glossaries
        if self.pretrained_entities_dimentionality_reduction is True:
            if args.entities_dimentionality_reduction is not None:
                self.dimred_e = self.fromnumpy(args.entities_dimentionality_reduction)
            else:
                self.dimred_e = self.fromnumpy(xavier_init((args.entities.shape[1], self.edim)))
        else:
            if args.entities_dimentionality_reduction is not None:
                self.dimred_e = self.tnnparameter(t.from_numpy(args.entities_dimentionality_reduction))
            else:
                self.dimred_e = self.tnnparameter(t.from_numpy(xavier_init((args.entities.shape[1], self.edim))))
        # dimensionality reduction, relations' feature  glossaries
        if self.pretrained_relations_dimentionality_reduction is True:
            if args.relations_dimentionality_reduction is not None:
                self.dimred_r = self.fromnumpy(args.relations_dimentionality_reduction)
            else:
                self.dimred_r = self.fromnumpy(xavier_init((args.relations.shape[1], self.rdim)))
        else:
            if args.entities_dimentionality_reduction is not None:
                self.dimred_r = self.tnnparameter(t.from_numpy(args.relations_dimentionality_reduction))
            else:
                self.dimred_r = self.tnnparameter(t.from_numpy(xavier_init((args.relations.shape[1], self.rdim))))
        # current ongoing entity/rel/ParamRegardClus to learn
        self.entity_current_id= 0
        self.relation_current_id= 0
        self.entity_cluster_current_id= 0
        self.relation_cluster_current_id= 0
        # define loss function for model:
        if self.projElossfcnType=='pointwise':
            self.projBfcntype= t.nn.functional.sigmoid
        elif self.projElossfcnType== 'listwise':
            self.projBfcntype= t.nn.functional.softmax


    def fromnumpy(self,dat):
        import torch as t
        if self.cudaenabled==True:
            return t.from_numpy(dat).cuda()
        else:
            return t.from_numpy(dat)


    def tnnparameter(self,tens):
        import torch as t
        if self.cudaenabled==True:
            return t.nn.Parameter(tens.cuda())
        else:
            return t.nn.Parameter(tens)

    def tensors2cuda(self):
        for itm in self.__dict__.keys():
            if type(self.__dict__[itm]) is t.Tensor:
                self.__dict__[itm]= self.__dict__[itm].cuda()

    def setmode(self,mode='train'):
        if mode=='train' or mode=='training':
            self.mode='train'
        elif mode=='test' or mode=='testing' or mode=='valid' or mode=='validation':
            self.mode=='test'


    def forward(self,input_headRelIds):
        from scipy import sparse
        # Only one instance (here 1x2 [relid,entid]) gets passed to the fun
        # input_headEntIds is a tensor of batch of selected head and its relation id
        self.relation_current_id=input_headRelIds[0][0].tolist()
        self.head_current_id=input_headRelIds[0][1].tolist()
        self.relation_current_cluster_id= self.relations_clusters_id[self.relation_current_id]  ##TODO: TEST IT VOCAB RIGHT OR WRONG
        self.head_current_cluster_id= self.entities_clusters_id[self.head_current_id]  ##TODO: TEST IT VOCAB RIGHT OR WRONG
        # get embeds for each:
        self.De = self.entities_embeds[self.head_current_id, :].clone()#self.De= self.fromnumpy((np.asarray(self.entities_embeds[self.head_current_id,:])[0]))
        self.Dr = self.relations_embeds[self.relation_current_id, :].clone()# self.Dr= self.fromnumpy((np.asarray(self.relations_embeds[self.relation_current_id,:])[0]))
        # get positive candidates, sample negative candidates: (positively samples all, negatively samples some)
        self.ps=self.tails_glossary_tensor[self.head_current_id, self.relation_current_id, :]
        self.psid= self.tails_glossary_tensor.where[self.head_current_id, self.relation_current_id, :]
        negative_cnt= self.ps.shape[1]-np.sum(self.ps!=0)
        probs4neg= np.random.random(self.ps.shape)
        self.ns= (sparse.lil_matrix(np.ones(self.ps.shape))-self.ps).multiply(probs4neg<self.negative_sampler_thresh).tolil()
        self.nsid= self.ns.nonzero()[1]
        # get params for each:
        self.e=self.tnnparameter(self.E[self.head_current_id,:].clone()) # ALSO THIS SUSTAINS OBJ w1.data -= learning_rate * w1.grad.data
        self.r=self.tnnparameter(self.R[self.relation_current_id,:].clone()) # ALSO THIS SUSTAINS OBJ w1.data -= learning_rate * w1.grad.data
        self.cr=self.tnnparameter(self.CR[ self.relation_current_cluster_id ,:].clone())
        self.ce=self.tnnparameter(self.CE[self.head_current_cluster_id,:].clone())
        # get (clus regul)params for each:
        if self.regularize_within_clusters_relations == True:
            ids4relationsclus=list(set(self.id2clustersid4relations[self.relation_current_cluster_id])-set([self.relation_current_cluster_id]))
            self.clusrelations= self.tnnparameter( self.R[ids4relationsclus,:].clone() )# except for the main entity
            self.clusrelations_plus_mainrelation = t.cat((self.clusrelations,self.r.unsqueeze(0)),dim=0)
            lossreguls1 = self.regularization_rate_relations * t.sqrt(t.sum(self.clusrelations_plus_mainrelation.var(0)))
        else:
            lossreguls1=t.tensor(0)
        if self.regularize_within_clusters_entities == True:
            ids4entitiesclus=list(set(self.id2clustersid4entities[self.head_current_cluster_id])-set([self.head_current_cluster_id]))
            self.clusentities= self.tnnparameter( self.E[ids4entitiesclus,:].clone() ) # except for the main entity
            self.clusentities_plus_mainentity = t.cat((self.clusentities,self.e.unsqueeze(0)),dim=0)
            lossreguls2= self.regularization_rate_entities * t.sqrt(t.sum(self.clusentities_plus_mainentity.var(0)))
        else:
            lossreguls2 = t.tensor(0)
        lossreguls= lossreguls1+lossreguls2
        # pass first layer of projB:
        lyr1= (self.De.unsqueeze(0).mm( self.dimred_e)*(self.e.unsqueeze(0)) + self.ce.unsqueeze(0)).t() * (self.Dr.unsqueeze(0).mm( self.dimred_r)*(self.r.unsqueeze(0)) + self.cr.unsqueeze(0))
        # pass activation function f : here tanh
        lyr1_act=t.tanh(lyr1)
        # pass next layer of projB:
        lyr2_pos= self.entities_embeds[self.psid,:].clone().mm( self.dimred_e).mm(lyr1_act.mm(self.r.unsqueeze(0).t())+self.bp.t())
        lyr2_neg= self.entities_embeds[self.nsid,:].clone().mm( self.dimred_e).mm(lyr1_act.mm(self.r.unsqueeze(0).t())+self.bp.t())
        # pass activation function g: here softmax
        lyr2_pos_act= self.projBfcntype(lyr2_pos,0)
        lyr2_neg_act= self.projBfcntype(lyr2_neg,0)
        # pass to margin-based loss function  # get cross entropy with its corresponding tails as outputs (supervised weighted labels)
        if self.mode == 'train':
            vec_of_relids_rep_neg= [self.relation_current_id for _ in range(len(self.nsid))]
            lossneg= -self.fromnumpy(self.ns[0,self.nsid]/(1+self.sumoverheads[vec_of_relids_rep_neg, self.nsid])/(1+self.sumovertails[self.relation_current_id, self.head_current_id]) ).mm(t.log(1-lyr2_neg_act+1e-9))
            vec_of_relids_rep_pos= [self.relation_current_id for _ in range(len(self.psid))]
            losspos= -self.fromnumpy(self.ps[0,self.psid]/(1+self.sumoverheads[vec_of_relids_rep_pos, self.psid])/(1+self.sumovertails[self.relation_current_id, self.head_current_id]) ).mm(t.log(lyr2_pos_act+1e-9))
            loss=lossneg+losspos+lossreguls

            # redefine the optimizer:
            self.opt=t.optim.SGD(self.parameters(),lr=0.05,momentum=0.3,weight_decay=self.weight_decay) ##TODO: Add args' tunable arguments
            self.opt.zero_grad()
            # backward the loss to compute gradients
            loss.backward()
            # update the data :
            self.update_entities_relations_parameters(self.head_current_id,self.relation_current_id,self.head_current_cluster_id,self.relation_current_cluster_id,ids4relationsclus,ids4entitiesclus)
            # clear unneeded things:
            ##TODO
        elif self.mode=='test':
            vec_of_relids_rep_neg= [self.relation_current_id for _ in range(len(self.nsid))]
            lossneg= -self.fromnumpy(self.ns[0,self.nsid]/(1+self.sumoverheads[vec_of_relids_rep_neg, self.nsid]) ).mm(t.log(1-lyr2_neg_act+1e-9))
            vec_of_relids_rep_pos= [self.relation_current_id for _ in range(len(self.psid))]
            losspos= -self.fromnumpy(self.ps[0,self.psid]/(1+self.sumoverheads[vec_of_relids_rep_pos, self.psid]) ).mm(t.log(lyr2_pos_act+1e-9))
            pass
        if self.cudaenabled==False:
            return [lossneg.data.numpy().squeeze(0).squeeze(0).tolist(),losspos.data.numpy().squeeze(0).squeeze(0).tolist(),lossreguls.data.numpy().squeeze(0).squeeze(0).tolist()] #[loss.data.numpy().squeeze(0).squeeze(0).tolist()]
        else:
            return [lossneg.data.cpu().numpy().squeeze(0).squeeze(0).tolist(),losspos.data.cpu().numpy().squeeze(0).squeeze(0).tolist(),lossreguls.data.cpu().numpy().squeeze(0).squeeze(0).tolist()] #[loss.data.cpu().numpy().squeeze(0).squeeze(0).tolist()]






    def forward_batch(self,input_headRelIds):
        from scipy import sparse
        # Only one instance (here 1x2 [relid,entid]) gets passed to the fun
        # input_headEntIds is a tensor of batch of selected head and its relation id
        input_headRelIds=np.asarray(input_headRelIds)
        self.relations_current_id=input_headRelIds[:,0].tolist()
        self.heads_current_id=input_headRelIds[:,1].tolist()
        self.relations_current_cluster_id= self.relations_clusters_id[self.relations_current_id]  ##TODO: TEST IT VOCAB RIGHT OR WRONG
        self.heads_current_cluster_id= self.entities_clusters_id[self.heads_current_id]  ##TODO: TEST IT VOCAB RIGHT OR WRONG
        # get embeds for each:
        self.De = self.entities_embeds[self.heads_current_id, :] #batch x heads
        self.Dr = self.relations_embeds[self.relations_current_id, :]# batch x rels
        # get positive candidates, sample negative candidates: (positively samples all, negatively samples some)
        ## get rels , heads and tails vecs ,, then save them FOR NEXT, combine head and tail to get 2nd indice of tnsr
        self.ps=self.tails_glossary_tensor[self.head_current_id, self.relation_current_id, :]
        self.psid= self.tails_glossary_tensor.where[self.head_current_id, self.relation_current_id, :]
        negative_cnt= self.ps.shape[1]-np.sum(self.ps!=0)
        probs4neg= np.random.random(self.ps.shape)
        self.ns= (sparse.lil_matrix(np.ones(self.ps.shape))-self.ps).multiply(probs4neg<self.negative_sampler_thresh).tolil()
        self.nsid= self.ns.nonzero()[1]
        # get params for each:
        self.e=self.tnnparameter(self.E[self.head_current_id,:].clone()) # ALSO THIS SUSTAINS OBJ w1.data -= learning_rate * w1.grad.data
        self.r=self.tnnparameter(self.R[self.relation_current_id,:].clone()) # ALSO THIS SUSTAINS OBJ w1.data -= learning_rate * w1.grad.data
        self.cr=self.tnnparameter(self.CR[ self.relation_current_cluster_id ,:].clone())
        self.ce=self.tnnparameter(self.CE[self.head_current_cluster_id,:].clone())
        # get (clus regul)params for each:
        if self.regularize_within_clusters_relations == True:
            ids4relationsclus=list(set(self.id2clustersid4relations[self.relation_current_cluster_id])-set([self.relation_current_cluster_id]))
            self.clusrelations= self.tnnparameter( self.R[ids4relationsclus,:].clone() )# except for the main entity
            self.clusrelations_plus_mainrelation = t.cat((self.clusrelations,self.r.unsqueeze(0)),dim=0)
            lossreguls1 = self.regularization_rate_relations * t.sqrt(t.sum(self.clusrelations_plus_mainrelation.var(0)))
        else:
            lossreguls1=t.tensor(0)
        if self.regularize_within_clusters_entities == True:
            ids4entitiesclus=list(set(self.id2clustersid4entities[self.head_current_cluster_id])-set([self.head_current_cluster_id]))
            self.clusentities= self.tnnparameter( self.E[ids4entitiesclus,:].clone() ) # except for the main entity
            self.clusentities_plus_mainentity = t.cat((self.clusentities,self.e.unsqueeze(0)),dim=0)
            lossreguls2= self.regularization_rate_entities * t.sqrt(t.sum(self.clusentities_plus_mainentity.var(0)))
        else:
            lossreguls2 = t.tensor(0)
        lossreguls= lossreguls1+lossreguls2
        # pass first layer of projB:
        lyr1= (self.De.unsqueeze(0).mm( self.dimred_e)*(self.e.unsqueeze(0)) + self.ce.unsqueeze(0)).t() * (self.Dr.unsqueeze(0).mm( self.dimred_r)*(self.r.unsqueeze(0)) + self.cr.unsqueeze(0))
        # pass activation function f : here tanh
        lyr1_act=t.tanh(lyr1)
        # pass next layer of projB:
        lyr2_pos= self.entities_embeds[self.psid,:].clone().mm( self.dimred_e).mm(lyr1_act.mm(self.r.unsqueeze(0).t())+self.bp.t())
        lyr2_neg= self.entities_embeds[self.nsid,:].clone().mm( self.dimred_e).mm(lyr1_act.mm(self.r.unsqueeze(0).t())+self.bp.t())
        # pass activation function g: here softmax
        lyr2_pos_act= self.projBfcntype(lyr2_pos,0)
        lyr2_neg_act= self.projBfcntype(lyr2_neg,0)
        # pass to margin-based loss function  # get cross entropy with its corresponding tails as outputs (supervised weighted labels)
        if self.mode == 'train':
            vec_of_relids_rep_neg= [self.relation_current_id for _ in range(len(self.nsid))]
            lossneg= -self.fromnumpy(self.ns[0,self.nsid]/(1+self.sumoverheads[vec_of_relids_rep_neg, self.nsid])/(1+self.sumovertails[self.relation_current_id, self.head_current_id]) ).mm(t.log(1-lyr2_neg_act+1e-9))
            vec_of_relids_rep_pos= [self.relation_current_id for _ in range(len(self.psid))]
            losspos= -self.fromnumpy(self.ps[0,self.psid]/(1+self.sumoverheads[vec_of_relids_rep_pos, self.psid])/(1+self.sumovertails[self.relation_current_id, self.head_current_id]) ).mm(t.log(lyr2_pos_act+1e-9))
            loss=lossneg+losspos+lossreguls
            # if self.cudaenabled:
            #     self = self.cuda()  #
            #     self.tensors2cuda()
            #     ###
            #     # redefine the optimizer:
            #     self.opt=t.optim.SGD([],lr=0.05,momentum=0.3,weight_decay=self.weight_decay) ##TODO: Add args' tunable arguments
            # else:
            #     self.opt=t.optim.SGD(self.parameters(),lr=0.05,momentum=0.3,weight_decay=self.weight_decay) ##TODO: Add args' tunable arguments
            # redefine the optimizer:
            self.opt=t.optim.SGD(self.parameters(),lr=0.05,momentum=0.3,weight_decay=self.weight_decay) ##TODO: Add args' tunable arguments
            self.opt.zero_grad()
            # backward the loss to compute gradients
            loss.backward()
            # update the data :
            self.update_entities_relations_parameters(self.head_current_id,self.relation_current_id,self.head_current_cluster_id,self.relation_current_cluster_id,ids4relationsclus,ids4entitiesclus)
            # clear unneeded things:
            ##TODO
        elif self.mode=='test':
            vec_of_relids_rep_neg= [self.relation_current_id for _ in range(len(self.nsid))]
            lossneg= -self.fromnumpy(self.ns[0,self.nsid]/(1+self.sumoverheads[vec_of_relids_rep_neg, self.nsid]) ).mm(t.log(1-lyr2_neg_act+1e-9))
            vec_of_relids_rep_pos= [self.relation_current_id for _ in range(len(self.psid))]
            losspos= -self.fromnumpy(self.ps[0,self.psid]/(1+self.sumoverheads[vec_of_relids_rep_pos, self.psid]) ).mm(t.log(lyr2_pos_act+1e-9))
            pass
        if self.cudaenabled==False:
            return [lossneg.data.numpy().squeeze(0).squeeze(0).tolist(),losspos.data.numpy().squeeze(0).squeeze(0).tolist(),lossreguls.data.numpy().squeeze(0).squeeze(0).tolist()] #[loss.data.numpy().squeeze(0).squeeze(0).tolist()]
        else:
            return [lossneg.data.cpu().numpy().squeeze(0).squeeze(0).tolist(),losspos.data.cpu().numpy().squeeze(0).squeeze(0).tolist(),lossreguls.data.cpu().numpy().squeeze(0).squeeze(0).tolist()] #[loss.data.cpu().numpy().squeeze(0).squeeze(0).tolist()]







    def update_entities_relations_parameters(self,entid,relid,entclid,relclid,ids4relationsclus,ids4entitiesclus):
        ##note: self.dimred_e self.dimred_r are automatically retained and learned and don't need this func
        # once grads are available, the parameters get updated:
        self.opt.step()
        # now all the changed data gets resaved back to the glossaries:
        # if self.cudaenabled==False:
        self.E.data[ids4entitiesclus,:]=self.clusentities.data # entity vecs to indirectly learn
        self.R.data[ids4relationsclus,:]=self.clusrelations.data # relation vecs to indirectly learn
        self.E.data[entid,:]=self.e.data # entity vecs to indirectly learn
        self.R.data[relid,:]=self.r.data # relation vecs to indirectly learn
        self.CE.data[entclid,:]=self.ce.data # Parameters of all ENTITIES' clusters to indirectly learn
        self.CR.data[relclid,:]=self.cr.data # Parameters of all RELATIONS' clusters to indirectly learn









