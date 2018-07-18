
file 1: SRL_preprocessing.py 
	
	It contains the entire preprocessing phase performed on the conl dataset,
	the data collected by it are saved in packets that are then 
	opened by the NN.

file 2: SRL_NN.py

	Contain the biLSTM used to train the model and to predict the the output values. 

file 3: Disambiguation Preprocessing

	Prepare Conl and SemCor dataset to be used by the neural network.

file 4: disambiguate Neural network

	Network used to disambiguate the dataset.

file 5: utils_functions.py

	Some auxiliares functions.

file 6: extension_3.py

	My attempt of define the "selectional preference of PropBank predicates
	and their arguments"
