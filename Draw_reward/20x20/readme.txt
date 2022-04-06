Gurobi:
        self.M_plus = 1000
        self.M_minus = -1000
        self.D_size = 40
        self.B_size = 50
RL:
	# The parameter interfere the training time
	BATCH_SIZE = 32
	MEMORY_CAPACITY = 100
	GAME_STEP_NUM = 199
	EPOCHS = 1000
	LR = 0.01  # learning rate
	EPSILON = 0.9  # greedy policy
	GAMMA = 0.9  # reward discount
	TARGET_REPLACE_ITER = 100  # target update frequency
	# Network parameters
	input_dim = 794
	output_dim = 1
	hidden_dim = 16

##  iters = 10400  ##