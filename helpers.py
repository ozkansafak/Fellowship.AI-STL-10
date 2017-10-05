import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import matplotlib.pylab as pylab
from matplotlib.ticker import MaxNLocator
from keras.callbacks import Callback

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def print_runtime(start):
    end = time.time()
    print('Runtime: %d min %d sec' % ((end-start)//60, (end-start)%60))
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def plotter(xlabel=None, ylabel=None, title=None, xlim=None, ylim=None):
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches((15,5))
    plt.grid('on')
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if xlim: plt.xlim((0, xlim));
    if ylim: plt.ylim((0, ylim));
    if title: plt.title(title)
    return fig, ax

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# HELPER FUNCTIONS FOR THE NEURAL NETWORK MODEL

class Callback_Func(Callback):
    def __init__(self, train_data, test_data, start, wanna_plot=True):
        self.train_data = train_data
        self.test_data = test_data
        self.loss_train = []
        self.loss_test = []
        self.acc = []
        self.start = start
        self.wanna_plot = wanna_plot
        
        
    def plotter(self, title='validation accuracy'):
        ax = plt.subplot(121)
        plt.xlabel('epochs')
        plt.grid('on')
        plt.title('loss')
        x_plot = range(1, len(self.loss_train)+1)
        plt.xlim((1,max(max(x_plot),2)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(x_plot, self.loss_test, 'r--^', alpha=.7, label="validation")
        ax.plot(x_plot, self.loss_train, 'k--^', alpha=.7, label="train")
        plt.legend()

        ax = plt.subplot(122)
        plt.xlabel('epochs')
        plt.grid('on')
        ax.plot(x_plot, self.acc, 'b--^', alpha=.5)
        plt.xlim((1,max(max(x_plot),2)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(title)
        plt.ion()
        plt.show()


    def on_epoch_end(self, epoch, logs={}):
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        _loss_train, _ = self.model.evaluate(X_train, y_train, verbose=0)
        _loss_test, _acc = self.model.evaluate(X_test, y_test, verbose=0)
        self.loss_train.append(_loss_train)
        self.loss_test.append(_loss_test)
        self.acc.append(_acc)
        
        
        if self.wanna_plot: 
            self.plotter()
            end = time.time()
            print('\nRuntime: %d min %d sec' % ((end-self.start)//60, (end-self.start)%60))
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_cnames_fidx():
    # Returns
    # fidx: fold indices on the training set (0,...4999)
    # cnames: class names (10 classes)
    
    with open('data/class_names.txt','r') as f:
        cnames = f.read()
    cnames = cnames.split('\n')

    with open('data/fold_indices.txt','r') as f:
        fidx = f.read()
    fidx = [row[:-1].split(' ') for row in fidx.split('\n')][:-1]

    for i in range(len(fidx)):
        for j in range(len(fidx[i])):
            fidx[i][j] = int(fidx[i][j])
        
    return cnames, fidx

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def npy_saver(arr, arr_name, sno, verbose=True):
    with open('data/%s_%02d.npy' % (arr_name, sno), 'wb+') as f:
        np.save(f, arr)
        if verbose:
            print('Save: %s_%02d.npy' % (arr_name, sno))
    return

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_model_top():
    model_top = Sequential() 
    model_top.add(GlobalAveragePooling2D(data_format='channels_last', input_shape=(7, 7, 512))) 
    # model_top.add(Flatten())
    # model_top.add(Dense(1000, activation='relu')) 
    model_top.add(Dense(1000, activation='softmax')) 

    model_top.summary() 

    model_top.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) 

    return model_top

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def train_model(model_top, 
                X_train, y_train,
                X_cv, y_cv,
                epochs=20,
                batch_size=128,
                wanna_plot=False,
                fpath='model'):
    
    start = time.time()
    from keras.callbacks import ModelCheckpoint  
    print('Initiate Training....')
    # ..................................................................
    
    callback_inst = Callback_Func((X_train, y_train),(X_cv, y_cv), start, wanna_plot=wanna_plot)

    checkpointer = ModelCheckpoint(filepath='saved_models/' + fpath + '.best.hdf5', 
                                   verbose=1, save_best_only=True)

    model_top.fit(X_train, y_train, 
              validation_data=(X_cv, y_cv),
              epochs=epochs, 
              batch_size=batch_size, 
              callbacks=[callback_inst, checkpointer], 
              verbose=0)

    print_runtime(start)

    return model_top, callback_inst









