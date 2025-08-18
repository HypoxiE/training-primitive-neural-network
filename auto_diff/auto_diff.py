import math

class Node:
	def __init__(self, val, parents=(), op="", label=None):
		self.grad = 0.0
		self.val = float(val)
		self.parents = parents
		self.op = op
		self.label = label

		self._backward = lambda: None

	def __repr__(self):
		name = f"{self.label}:" if self.label else ""
		return f"<{name}{self.op or 'var'} val={self.val:.6g} grad={self.grad:.6g}>"
	
	@staticmethod
	def _to_node(x):
		return x if isinstance(x, Node) else Node(x)
	
	def __add__(self, other):
		other = Node._to_node(other)
		out = Node(self.val + other.val, parents=[(self, lambda g: g), (other, lambda g: g)], op="+")

		def _backward():
			self.grad += out.grad * 1.0
			other.grad += out.grad * 1.0

		out._backward = _backward
		return out
	
	def __radd__(self, other): return self + other

	def __neg__(self):
		out = Node(-self.val, parents=[(self, lambda g: -g)], op="neg")
		def _backward():
			self.grad += -out.grad
		out._backward = _backward
		return out
	
	def __sub__(self, other): return self + (-other)
	def __rsub__(self, other): return Node._to_node(other) + (-self)

	def __mul__(self, other):
		other = Node._to_node(other)
		out = Node(self.val * other.val, parents=[(self, lambda g: g * other.val), (other, lambda g: g * self.val)], op="*")
		def _backward():
			self.grad += out.grad * other.val
			other.grad += out.grad * self.val
		out._backward = _backward
		return out
	
	def __rmul__(self, other): return self * other

	def __truediv__(self, other):
		other = Node._to_node(other)
		out = Node(self.val / other.val, parents=[(self, lambda g: g / other.val), (other, lambda g: -g * self.val / (other.val**2))], op="/")
		def _backward():
			self.grad += out.grad / other.val
			other.grad += -out.grad * self.val / (other.val**2)
		out._backward = _backward
		return out
	
	def __rtruediv__(self, other): return Node._to_node(other) / self

	def __pow__(self, p: float):
		# степень фиксируем скаляром p
		out = Node(self.val**p, parents=[(self, lambda g: g * p * (self.val ** (p-1)))], op=f"**{p}")
		def _backward():
			self.grad += out.grad * p * (self.val ** (p-1))
		out._backward = _backward
		return out
	
	def __rpow__(self, p: float):
		# степень фиксируем скаляром p
		out = Node(p**self.val, parents=[(self, lambda g: g * p ** self.val * math.log(p))], op=f"{p}**")
		def _backward():
			self.grad += out.grad * p ** self.val * math.log(p)
		out._backward = _backward
		return out

def backward(loss: Node):
	# 1) топологическая сортировка (DFS)
	topo, visited = [], set()
	def build(u: Node):
		if u not in visited:
			visited.add(u)
			for p, _ in u.parents:
				build(p)
			topo.append(u)
	build(loss)
	print(topo, list(reversed(topo)), visited)
	# 2) инициализируем dL/dL = 1 и идём в обратном порядке
	for n in topo:
		n.grad = 0.0
	loss.grad = 1.0
	for node in reversed(topo):
		node._backward()
		print(node)

if __name__ == "__main__":

	x = Node(2, label="x")
	y = Node(3, label="y")
	print(x)
	f = 8**x
	print(f)
	backward(f)
	print(x.grad)
