import os
from pathlib import Path
import kagglehub
import pandas as pd
import sys
import pyperclip

def download_data():
    ''' 
    downloads new data for kaggle competition 
    and creates new directory for project 
    '''

    if input('is working directory set to "kaggle" ? [y/n] ') != 'y':
        print('set working directory, then rerun')
        exit()

    # copy api key from env vars for login
    pyperclip.copy(os.getenv('KAGGLE_KEY'))
    print('copied API token')
    kagglehub.login()

    # enter competition name
    if len(sys.argv) <= 1:
        print('competition name from url : https://www.kaggle.com/competitions/comp-name/data')
        comp_name = input('enter competition name: ')
    else:
        comp_name = sys.argv[1].lower().strip()
    
    # create new dir and set
    try:
        os.mkdir(comp_name)
        print('created new project directory:', comp_name)
    except FileExistsError:
        pass
    local_dir = os.path.join(os.getcwd(), comp_name)
    os.chdir(local_dir)
    # print('set directory:', local_dir)

    # download data bundle
    kaggle_dir = Path(kagglehub.competition_download(comp_name))
    print('loaded data to:', kaggle_dir)
    files = sorted(p.name for p in kaggle_dir.iterdir())

    # save files for use in project
    print('saving files to', os.getcwd())
    for f in files:
        try:
            data = pd.read_csv(kaggle_dir / f)
            data.to_csv(f)
        except pd.errors.ParserError:
            with open(kaggle_dir / f, 'r') as kagglefile:
                data = kagglefile.read()
            with open((f), 'w+') as newfile:
                    newfile.write(data)
        print('downloaded', f)

if __name__ == '__main__':
    download_data()