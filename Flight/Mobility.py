def mobility(A,B,X_Before,X_After)
	'''
	A = [a_1,a_2,a_3,...,a_n]
	B = [b_1,b_2,b_3,...,b_4]
	X_Before = [X_before_1,X_before_2,...,X_before_n]
	
	Estimate
	X_before_{t} = a_t * Alpha + X_before{t-1} * T - b_t * Beta
	
	# Notice , We do not know a_t and b_t if t is in predict time period
	# So Without this information , this model can be used to reveal the machenism of the mobility but it's hard to do pedict
	
	Optimization
	Argmin \sum_{i=1}^{k} (|| a * Alpha_{i} + X_{t-1} * (I + T) - b * Beta_{i} - X_{t} ||)_{2}^{2} + || Alpah||^2 + ||Beta||^2 + ||T||^2
	'''
	#Initial Coef
	Alpha = 
	Beta = 
	T = 
	
	#Function
	def Norm(X):
		pass
	
	#Update Coef
	for a , b , X_before , X_after in zip(A,B,X_Before,X_After):
		for i in range(len(Alpha)):
			Diff_Alpha_i = ( a * Alpha[i] + X_before[i] + T[i,:] * X_before[i] - b * Beta[i] - X_after[i] ) * a + Gamma * Alpha[i]
			Alpha[i] = Alpha[i] - Lambda * Diff_Alpha_i
			
			Diff_Beta_i = ( a * Alpha[i] + X_before[i] + T[i,:] * X_before[i] - b * Beta[i] - X_after[i] ) * b + Gamma * Beta[i]
			Beta[i] = Beta[i] - Lambda * Diff_Beta_i
			
			Diff_T_i = ( a * Alpha[i] + X_before[i] + T[i,:] * X_before[i] - b * Beta[i] - X_after[i] ) * X_before.T + Gamma * T[i,:]
			T[i,:] = T[i,:] - Lambda * Diff_T_i
			
