CS 725 FML Assignment-2

Deepak Singh Baghel  203050005  203050005@iitb.ac.in

NOTE:
	For LR=0.01 Layers=2 units=64 lambda=5  train and dev rmse both were nan because of exploding weights 
	hence for this setting only np.clip(w, -1k, 1k) method is used to bring down the heavy weights rest of the 
	results	are calculated without using np.clip(). Since weights in NN are in range -2 to 2 generally 
	hence clipping the heavy weights in range -1k, 1k does not affect actual result.
