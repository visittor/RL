def IDGenerator( )-> int:
	i = int(0)
	while True:
		yield i
		i += 1