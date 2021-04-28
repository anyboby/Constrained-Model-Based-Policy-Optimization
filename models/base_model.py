import abc

class BaseModel(abc.ABC):

    @abc.abstractmethod
    def predict(self, x):
        """ Make predictions, should return (mean, var) if model is probabilistic or mean else"""
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, x, y, ):
        """ Make predictions, should return (mean, var) if model is probabilistic or mean else"""
        raise NotImplementedError

    @abc.abstractproperty
    def is_probabilistic(self):
        """ indicates whether model predictions are probabilistic or deterministic """
        raise NotImplementedError 

    @abc.abstractproperty
    def is_ensemble(self):
        """ indicates whether model is an ensemble """
        raise NotImplementedError 

    @abc.abstractproperty
    def in_dim(self):
        """ dimension of inputs """
        raise NotImplementedError 

    @abc.abstractproperty
    def out_dim(self):
        """ dimension of outputs """
        raise NotImplementedError 

class EnsembleModel(BaseModel):
    @abc.abstractmethod
    def predict_ensemble(self, x):
        """ Make predictions of whole ensemble, output shape should be (ensemble, batch_size, y_shape)"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def elite_inds(self,):
        """ Returns indices of the elite models"""
        raise NotImplementedError