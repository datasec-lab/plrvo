class DataCollector:
  def __init__(self):
    self.data = []
    
  def entry(self, accuracy = '*', epochs = '*', learning_rate = '*', epsilon = '*', delta = '*', clipping_threshold = '*', 
            sampling_rate = '*', batch_size = '*', training_time = '*', **kwargs
           ):
    import pandas as pd
    #df1 = pd.DataFrame.from_dict(args)
    if len(kwargs) > 0:
        print(kwargs)
        df2 = pd.DataFrame.from_dict(kwargs)
    d = {'Accuracy':accuracy, 'Epochs':epochs, 'Learning Rate':learning_rate, 
         'Epsilon':epsilon, 'Delta':delta, 'Clipping Threshold':clipping_threshold, 
         'Sampling Rate':sampling_rate, 'Batch Size': batch_size, 'Training Time':training_time
        }
    df3 = pd.DataFrame.from_dict(d)
    result = pd.concat([df3], axis=1)
    
  def save_table(self, filename ='table.csv'):
    pd.DataFrame.from_dict(self.data).to_csv(filename)
    
  def head(self, count=5):
    pd.DataFrame.from_dict(self.data).head()
    
  def add_values_post(self, **kwargs):
    for k in kwargs.keys():
        data[k].replace('*', kwargs[k], inplace = True)        
