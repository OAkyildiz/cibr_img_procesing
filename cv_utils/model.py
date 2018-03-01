import os
import pandas as pd
import numpy as np

class Model(object):
    def __init__(self,columns = ['x', 'y','label'], infile="grasp_labels", outfile = None):
        self.columns=columns
        self.path="data/"+infile

        self.init_data()
        if outfile == None:
            self.out=self.path
        else:
            self.out="data/"+outfile
            self.is_new=True

        self.x_limit,self.y_limit = -1, -1
        self.Mu, self.Sigma = -1, -1

    def init_data(self):
        print("Attempting to read "+self.path+".csv ...")
        try:
            self.data=pd.read_csv(self.path+".csv" ,na_values=["", " ", "-"])
            self.is_new=False
        except Exception, IOError:
            print("Couldn't read a file")
            self.data=pd.DataFrame(columns=self.columns)
            self.is_new=True
        self.results=pd.DataFrame(columns=self.columns)

    def save_data(self):
        print("Saving to read "+self.path+".csv ...")
        x=self.data.to_csv(self.out+".csv", mode='w', header=self.columns)
        print x
        if self.is_new:
            r = open()
            self.results.to_csv(self.out+"_results.csv", mode='w', header=self.columns)
        else:
            self.results.to_csv(self.out+"_results.csv", mode='a', header=self.False)

        f.close()
        r.close()

    def add_data(self, data_row):
        print("Adding", data_row)
        self.data=self.data.append(pd.Series(data_row,index=self.columns),ignore_index=True)
        self.update_params()

    def add_result(self,data_row):
        self.results=self.results.append(pd.Series(data_row,index=self.columns),ignore_index=True)


    def update_params(self):
        d=self.data
        invalid_i = d['label'] == -1 #there shoudl be a label
        square_i =  d['label'] == 1

        self.x_limit = min(d[invalid_i][self.columns[0]]) if invalid_i.any() else -1
        self.y_limit = min(d[invalid_i][self.columns[1]]) if invalid_i.any() else -1

        m=d[square_i][self.columns[1]]/d[square_i][self.columns[0]]#slopes #probably use lambda
        #print m
        self.Mu = np.mean(m)
        self.Sigma = np.std(m)

    def classify(self, x, y):

        if x<self.x_limit and y<self.y_limit:
            res = ( abs(x/y -self.Mu) < 2*self.Sigma )
        else:
            res = -1
        return res

    #BC pandas.append is gross
    #def append(self,df,data_row):
    #    df=df.append(pd.Series(data_row,index=self.columns),ignore_index=True)
    # well, this copies the object so we can't really pass it around  (like a cv2.Mat)

if __name__ == '__main__':
    print("This is jsut a module")
    sys.exit()
