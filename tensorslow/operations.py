import numpy as np
from tensorslow.graph import Operation


class add(Operation):
    """Returns x + y element-wise.
    """

    def __init__(self, x, y):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the add operation

        Args:
          x_value: First summand value
          y_value: Second summand value
        """
        self.inputs = [x_value, y_value]
        return x_value + y_value

    def grad(self, grad):
        """Computes the gradients for `add`.

        Args:
          grad: Gradient with respect to the output of the `add`.

        Returns:
          Gradients with respect to the input of `add`.
        """
        a = self.inputs[0]
        b = self.inputs[1]

        grad_wrt_a = grad
        while np.ndim(grad_wrt_a) > len(a.shape):
            grad_wrt_a = np.sum(grad_wrt_a, axis=0)
        for axis, size in enumerate(a.shape):
            if size == 1:
                grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)

        grad_wrt_b = grad
        while np.ndim(grad_wrt_b) > len(b.shape):
            grad_wrt_b = np.sum(grad_wrt_b, axis=0)
        for axis, size in enumerate(b.shape):
            if size == 1:
                grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)

        return [grad_wrt_a, grad_wrt_b]


class matmul(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b):
        """Construct matmul

        Args:
          a: First matrix
          b: Second matrix
        """
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        """Compute the output of the matmul operation

        Args:
          a_value: First matrix value
          b_value: Second matrix value
        """
        self.inputs = [a_value, b_value]
        return a_value.dot(b_value)

    def grad(self, grad):
        """Computes the gradients for `matmul`.

        Args:
          grad: Gradient with respect to the output of the `matmul`.

        Returns:
          Gradients with respect to the input of `matmul`.
        """

        A = self.inputs[0]
        B = self.inputs[1]

        return [grad.dot(B.T), A.T.dot(grad)]


class sigmoid(Operation):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self, a):
        """Construct sigmoid

        Args:
          a: Input node
        """
        super().__init__([a])

    def compute(self, a_value):
        """Compute the output of the sigmoid operation

        Args:
          a_value: Input value
        """
        return 1 / (1 + np.exp(-a_value))

    def grad(self, grad):
        """Computes the gradients for `sigmoid`.

        Args:
          grad: Gradient with respect to the output of the `sigmoid`.

        Returns:
          Gradients with respect to the input of `sigmoid`.
        """

        sigmoid = self.output

        return grad * sigmoid * (1 - sigmoid)


class softmax(Operation):
    """Returns the softmax of a.
    """

    def __init__(self, a):
        """Construct softmax

        Args:
          a: Input node
        """
        super().__init__([a])

    def compute(self, a_value):
        """Compute the output of the softmax operation

        Args:
          a_value: Input value
        """
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=1)[:, None]

    def grad(self, grad):
        """Computes the gradients for `softmax`.

        Args:
          grad: Gradient with respect to the output of the `softmax`.

        Returns:
          Gradients with respect to the input of `softmax`.
        """

        softmax = self.output
        return (grad - np.reshape(
            np.sum(grad * softmax, 1),
            [-1, 1]
        )) * softmax


class log(Operation):
    """Computes the natural logarithm of x element-wise.
    """

    def __init__(self, x):
        """Construct log

        Args:
          x: Input node
        """
        super().__init__([x])

    def compute(self, x_value):
        """Compute the output of the log operation

        Args:
          x_value: Input value
        """
        return np.log(x_value)

    def grad(self, grad):
        """Computes the gradients for `log`.

        Args:
          grad: Gradient with respect to the output of the `log`.

        Returns:
          Gradients with respect to the input of `log`.
        """
        x = self.inputs[0]
        return grad/x


class multiply(Operation):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y):
        """Construct multiply

        Args:
          x: First multiplicand node
          y: Second multiplicand node
        """
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the multiply operation

        Args:
          x_value: First multiplicand value
          y_value: Second multiplicand value
        """
        return x_value * y_value

    def grad(self, grad):
        """Computes the gradients for `multiply`.

        Args:
          grad: Gradient with respect to the output of the `multiply` .

        Returns:
          Gradients with respect to the input of `multiply`.
        """

        A = self.inputs[0]
        B = self.inputs[1]

        return [grad * B, grad * A]


class reduce_sum(Operation):
    """Computes the sum of elements across dimensions of a tensor.
    """

    def __init__(self, A, axis=None):
        """Construct reduce_sum

        Args:
          A: The tensor to reduce.
          axis: The dimensions to reduce. If `None` (the default), reduces all dimensions.
        """
        super().__init__([A])
        self.axis = axis

    def compute(self, A_value):
        """Compute the output of the reduce_sum operation

        Args:
          A_value: Input tensor value
        """
        return np.sum(A_value, self.axis)

    def grad(self, grad):
        """Computes the gradients for `reduce_sum`.

        Args:
          grad: Gradient with respect to the output of the `reduce_sum`.

        Returns:
          Gradients with respect to the input of `reduce_sum`.
        """
        A = self.inputs[0]

        output_shape = np.array(A.shape)
        output_shape[self.axis] = 1
        tile_scaling = A.shape // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)


class negative(Operation):
    """Computes the negative of x element-wise.
    """

    def __init__(self, x):
        """Construct negative

        Args:
          x: Input node
        """
        super().__init__([x])

    def compute(self, x_value):
        """Compute the output of the negative operation

        Args:
          x_value: Input value
        """
        return -x_value

    def grad(self, grad):
        """Computes the gradients for `negative`.

        Args:
          grad: Gradient with respect to the output of the `negative`.

        Returns:
          Gradients with respect to the input of `negative`.
        """
        return -grad
