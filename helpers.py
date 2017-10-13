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
        _loss_train, _ = self.model.evaluate(X_train, y_train, batch_size=1024, verbose=0)
        _loss_test, _acc = self.model.evaluate(X_test, y_test, batch_size=1024, verbose=0)
        self.loss_train.append(_loss_train)
        self.loss_test.append(_loss_test)
        self.acc.append(_acc)
        
        
        if self.wanna_plot: 
            self.plotter()
            end = time.time()
            print('\nRuntime: %d min %d sec' % ((end-self.start)//60, (end-self.start)%60))
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_cnames_fidx(y_onehot, n_split=5):
    # Returns
    # fidx: fold indices on the training set (0,...4999)
    # cnames: class names (10 classes)
    
    with open('data/class_names.txt','r') as f:
        cnames = f.read()
    cnames = cnames.split('\n')[:-1]
    cnames = np.asarray(cnames)

    fidx = custom_KFold(y_onehot, n_split=n_split)
    
    print('Class names retrieved in cnames')
    print('K-Fold indices generated, n_split = %i'%(n_split))
    print('Fold width = %i'%(fidx.shape[1]))
    print('Number of data points for each class = %i'%(fidx.shape[1]/n_split))
    return cnames, fidx

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def npy_saver(arr, arr_name, sno, verbose=True):
    with open('data/%s_%02d.npy' % (arr_name, sno), 'wb+') as f:
        np.save(f, arr)
        if verbose:
            print('Save: %s_%02d.npy' % (arr_name, sno))
    return


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def construct_model():
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from keras.layers import Dropout, Flatten, Dense
    from keras.models import Sequential

    model_top = Sequential() 
    model_top.add(GlobalAveragePooling2D(data_format='channels_last', input_shape=(7, 7, 512))) 
    # model_top.add(Dense(1000, activation='relu')) 
    model_top.add(Dense(1000, activation='relu')) 
    model_top.add(Dense(1000, activation='softmax')) 


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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def custom_KFold(y_onehot, n_split=5):
    y = np.argmax(y_onehot, axis=1)
    # input y: numpy array of class ID's
    
    idx = []
    for cls in set(y):
        # idx[cls] is the indices to class no cls
        idx.append(np.where(y==cls)[0])
    idx = np.asarray(idx)
    for i in range(len(idx)):
        np.random.shuffle(idx[i])

    fidx = []
    fold_width = int(idx.shape[1]/n_split)
    for k in range(n_split):
        out = []
        for i in range(len(idx)):
            out.append(idx[i,k*fold_width : (k+1)*fold_width])
        out = np.concatenate(out)
        np.random.shuffle(out)
        fidx.append(out)
    fidx = np.array(fidx)
    np.random.shuffle(fidx)
    return fidx





