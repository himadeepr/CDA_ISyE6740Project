TO RUN THE CODES DOWNLOAD all files and store them in a single directory and change the working directory

1) Data folder has relevant data files and fbb3.sqldb which is the SQL database extracted.
2) Scripts folder has four scripts. 
	Run features.R first to extract features and obtain training and testing dataset. 
	features.R script will source init.R script
	models.R script has the base models, with 10-fold crossvalidation, SMOTE and stacking approach
		note-this file will run for approximately 3 hours
	2stageada.R has the implementation of the two stage model and results for all runs
		note - this file will run approximately for 7 hours
3) Results folder has the objects obtained at various stages so that they can be loaded easily without re-running the entire code again
4) Report and PPT include proposal, and PPT - Final Report is mailed