def IDGenerator( )-> int:
	i = int(0)
	while True:
		yield i
		i += 1

def compareShape( shape1, shape2 ):
	'''
		Compare if shape1 and shape2 are equal. -1 mean it is a wildcard.
	'''
	for s1, s2 in zip( shape1, shape2 ):
		if s1 != s2 and s1 != -1 and s2 != -1:
			return False
	
	return True