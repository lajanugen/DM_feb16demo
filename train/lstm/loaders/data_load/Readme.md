
Usage 
  
####  
require 'dialogs'  
local a = dialogs()  
batch,mask = a:get_next_batch(i) -- i = 1,2,3 for train,valid,test respectively   
####  
  
batch is a (#acts + #slots) x (length of dialog) tensor  
mask is a tensor of length (length of dialog). 1,0 indicate system and user utterance, respectively  

Data:  
The data is available in /home/llajan/DM/data  
The files in config (flist) specify which instances correspond to train, valid and test  
Train and validation instances are in directory train and test instances are in directory test  

dialogs.lua directly reads from this data directory, so no code changes are necessary in using it  

The format of the debug_print file is as follows:  

#####################################################  
Turn number           Utterance   
####   
The dialog act, slot, value representation in json  
A sequence of  
	* Act act_number  
	  The extracted dialog act   
	  Corresponding slots in a table  
	----------------------------  
	Tensor: Binary encoding of act + slot   
	Acts: Set of (act whether_present)  
	Slots: Set of (slot whether_present)   
#####################################################  
