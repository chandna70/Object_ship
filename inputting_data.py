##Library

import os,shutil,boto3
import regex as re
import pandas as pd
import numpy as np



class setting_up_data:
    def __init__(self, path_folder, next_dir='') :
        
        ##Initialization of location
        
        """
        dir_folder = location of folder
        next_dir= next location of directory
        """
        
        self.dir_folder=path_folder
        self.next_dir=next_dir
        
    def changeDir(self):
        
        ##change directory and get its directory
        if os.getcwd()!= self.dir_folder:
            expected_directory=self.dir_folder+'/'+self.next_dir
            os.chdir(expected_directory)
        else:
            print("You are already in the expected directory.")
        return os.getcwd()
    
    def get_size(self, to_dir ,filename, unit='bytes'):
        
        ##Reading all the bytes files
        os.chdir(self.changeDir()+'/'+to_dir)
        file_size = os.path.getsize(filename)
        exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
        if unit not in exponents_map:
            raise ValueError("Must select from \
            ['bytes', 'kb', 'mb', 'gb']")
        else:
            size = file_size / 1024 ** exponents_map[unit]
        return round(size, 3)   
    
    def count_data(self):
        
        ##Counting all the files
        
        path=self.changeDir()
        dict_obj=dict.fromkeys(os.listdir(path))
        for id in os.listdir(path):
            if id.endswith(".csv"):
              continue
            else:
                dict_obj[id]=len(os.listdir(path+'/'+id))
        return dict_obj
    
    def collecting_image_data (self, make_dir_folder,next_path,exclussive_folder):
        
        #Gathering from different locations folder into one folder directory
        self.changeDir()
        if not os.path.isdir(make_dir_folder):
            os.mkdir(make_dir_folder)
        else:
            print('YOUR NAME FOLDER HAS BEEN CREATED')
            return
        if len(os.listdir(make_dir_folder))==0:
            print("This Folder still empty")
        else:
            print('Number of the file:',len(os.listdir(os.getcwd())))
        mkdir_path=self.changeDir()+'/'+make_dir_folder
        base_path=self.changeDir()+'/'+next_path
        obj_list=[name_path for name_path in os.listdir(base_path) if name_path in exclussive_folder]
    
        print('YOUR CURRENT DIRECTORY:',os.getcwd())
        print('\nFolders which want to collecting all their file into one folder:',obj_list)
        for dir in obj_list:
            print("Copying all the",dir,"files to folder ",make_dir_folder)
            for name in os.listdir(base_path+'/'+dir):
                if name.endswith(".jpg") or name.endswith(".pdf") :
                    shutil.copy(base_path+"/"+dir+"/"+name
                            ,mkdir_path)
                print("Success Copy:",name,'\n')
        
        
    def seperate_new_dataset(self, filename, new_name):
        
        #If the DataFrame consists of jointed value without seperate space or character

        self.changeDir()
        if (filename.endswith(".csv")) or filename.endswith('.txt'):
            
            df_temp=pd.read_csv(filename,encoding= 'unicode_escape')
            df_temp.columns=[name.split(' ')[0] for name in df_temp.columns]
            if not os.path.isfile(new_name):
                return df_temp.to_csv(new_name,index=False) 
            else:
                print('YOUR FILE NAME HAS BEEN CREATED:',new_name)
                return
        else:
            raise ValueError("Must tabular data (csv, txt, xlsx)")
    
    def make_folder(self,name_folder=''):
        if not os.path.isdir(name_folder):
            os.makedirs(name_folder)
        else:
            print('YOUR FOLDER NAME HAS BEEN CREATED:',name_folder)
            
    def collect_files(self, ls_folder):
        
        #Collect all the files within folder and record into the dataframe
        ls_filepath=[]
        for i in ls_folder:
            path_join=os.path.join(os.getcwd(),i)
            if os.path.isdir(path_join):
                for j in os.listdir(path_join):
                    path_file=os.path.join(path_join,j)
                    ls_filepath.append(path_file)
            else:
                continue
        df_collect=pd.DataFrame(data=ls_filepath,columns=["path_sounds"])
        return df_collect

    