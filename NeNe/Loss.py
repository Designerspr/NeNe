'''providing 2 useful loss function:
    CEL(cross-entropy error)
    MSE(mean squared error)
for different uses in regression and classification.
'''
import numpy as np

class CEL(object):
    '''cross-entropy error should be used only in classification problems.
    Or it may cause errors when calculating log().
    '''

    def __init__(self):
        return
    @classmethod
    def get_loss(cls,y_output,y_target,return_accu=True):
        '''Return average CEL.
        
        Arguments:
            y_output, y_target
        
        Keyword Arguments:
            return_accu {bool} -- [description] (default: {True})
        
        Returns:
            [type] -- [description]
        '''

        CEL=-(np.log(y_output)*y_target)
        ACE=np.sum(CEL/len(y_output))
        if return_accu:
            match_num=0
            for (i,sample) in enumerate(y_output):
                y_predict=(np.max(sample)==sample)
                # if match
                if np.sum(np.abs((y_predict-y_target[i])))==0:
                    match_num +=1
            accu=match_num/len(y_output)
            return ACE,accu
        return ACE
    @classmethod
    def get_loss_deriv(cls,y_output,y_target):
        return y_target/y_output

class MSE(object):
    '''MSE is usually used in regression.  
    Also can be used in classification problems, but seems not perform that well.
    '''

    def __init__(self):
        return
    @classmethod
    def get_loss(cls,y_output,y_target,return_accu=True):
        '''Return average MSE.
        
        Arguments:
            y_output, y_target
        
        Keyword Arguments:
            return_accu {bool} -- [description] (default: {True})
        
        Returns:
            [type] -- [description]
        '''

        MSE=1/2*np.sum((y_output-y_target)**2)
        if return_accu:
            match_num=0
            for (i,sample) in enumerate(y_output):
                y_predict=(np.max(sample)==sample)
                # if match
                if np.sum(np.abs((y_predict-y_target[i])))==0:
                    match_num +=1
            accu=match_num/len(y_output)
            return MSE,accu
        return MSE
    @classmethod
    def get_loss_deriv(cls,y_output,y_target):
        return (y_output-y_target)