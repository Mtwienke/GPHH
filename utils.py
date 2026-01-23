#!/usr/bin/env python3

# utils.py
import math
import os
from datetime import datetime

def protected_div(x, y):
	"""Protected division to avoid ZeroDivisionError."""
	try:
		return x / y if abs(y) > 1e-6 else 1.0
	except Exception:
		return 1.0
	
	
def euclidean_distance(p1, p2):
	return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def mean(values):
	return sum(values) / len(values) if values else 0.0

def tree_to_infix(expr):
	stack = []
	for token in expr:
		if token in FUNCTIONS:
			b = stack.pop()
			a = stack.pop()
			stack.append(f"({a} {token} {b})")
		else:
			stack.append(token)
	return stack[0] if stack else "<invalid>"

class Logger:
	def __init__(self, filename=None, enabled=True):
		timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		self.filename = filename or f"gp_log_{timestamp}.txt"
		self.enabled = enabled
		os.makedirs("logs", exist_ok=True)
		self.file = open(os.path.join("logs", self.filename), 'a')
		
	def log(self, *args, sep=' ', end='\n'):
		if not self.enabled:
			return
		timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
		message = sep.join(str(arg) for arg in args)
		self.file.write(f"{timestamp} {message}{end}")
		self.file.flush()  # Ensure it's written immediately
		
	def close(self):
		if self.file:
			self.file.close()