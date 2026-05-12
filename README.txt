--- CREATE ENV
set wd to "Kaggle"
	
	cd C:\Users\nbrock\OneDrive - Hayden Homes\Documents\python\kaggle

	conda env create -f environment.yml


--- DOWNLOAD KAGGLE DATA
create a programmatic access key on Kaggle.com

	conda env config vars set KAGGLE_KEY=xxxxxxxx

set wd to new project directory under "Kaggle"

	cd projectname

download data for comp-name (found in url : https://www.kaggle.com/competitions/comp-name/data)

	py -m loadkaggle.py comp-name


