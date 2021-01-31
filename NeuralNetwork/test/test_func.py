from RL import BasicFunction as BF
from RL import checkGrad

import numpy as np

# a = np.random.randn( 3, 4 )
# b = np.random.randn( 4, 1 )
# checkGrad( BF.MatMul, a, b )

# a = np.random.randn( 3, 3, 4 )
# b = np.random.randn( 4, 1 )
# checkGrad( BF.MatMul, a, b )

# a = np.random.randn( 3, 4 )
# b = np.random.randn( 3, 4 )
# checkGrad( BF.Multiply, a, b )

# a = np.random.randn( 3, 4 )
# b = np.random.randn( 4 )
# checkGrad( BF.Multiply, a, b )

# a = np.random.randn( 6, 4 )
# b = np.random.randn( 6, 4 )
# checkGrad( BF.Add, a, b )

# a = np.random.randn( 6, 4 )
# b = np.random.randn( 4 )
# checkGrad( BF.Add, a, b )

# a = np.random.randn( 10, 11 )
# b = np.random.randn( 10, 11 )
# checkGrad( BF.Max, a, b )

# a = np.arange( 5 ) / 10
# b = .2
# checkGrad( BF.Max, a, b )

# a = np.arange( 4, dtype=np.float64 ).reshape( 2, 2 )
# b = np.ones	( 2 )
# checkGrad( BF.Max, a, b )

# a = np.random.randn( 10, 11 )
# b = np.random.randn( 10, 11 )
# checkGrad( BF.Divide, a, b )

# a = np.random.randn( 10, 11 )
# b = np.random.randn( 11 )
# checkGrad( BF.Divide, a, b )

# a = np.random.randn( 10, 11 )
# checkGrad( BF.Exp, a )

a = np.random.randn( 10, 11 )
checkGrad( BF.ReduceSum, a, 1 )
