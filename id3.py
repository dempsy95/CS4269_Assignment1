import re
import sys
import math
from random import *
from HRData import HRData

training_error = []
test_error = []
origin_examples = []
tests = []
root = None

class Tree(object):
	def __init__(self):
		self.is_leaf = True
		self.children = {}
		self.attribute = None

def import_data():
	"""
	* import data
		* examples are all the examples used for training
		* attributes are a list of attributes that each example contains
	"""
	attributes = {"Outlook" : {"Sunny", "Overcast", "Rain"}, "Temp" : {"Hot", "Mild", "Cool"}, "Humidity" : {"High", "Normal"}, "Wind" : {"Strong", "Weak"}}
	data = [None] * 14
	data[0] = {"Outlook" : "Sunny", "Temp" : "Hot", "Humidity" : "High", "Wind" : "Weak", "label" : "No"}
	data[1] = {"Outlook" : "Sunny", "Temp" : "Hot", "Humidity" : "High", "Wind" : "Strong", "label" : "No"}
	data[2] = {"Outlook" : "Overcast", "Temp" : "Hot", "Humidity" : "High", "Wind" : "Weak", "label" : "Yes"}
	data[3] = {"Outlook" : "Rain", "Temp" : "Mild", "Humidity" : "High", "Wind" : "Weak", "label" : "Yes"}
	data[4] = {"Outlook" : "Rain", "Temp" : "Cool", "Humidity" : "Normal", "Wind" : "Weak", "label" : "Yes"}
	data[5] = {"Outlook" : "Rain", "Temp" : "Cool", "Humidity" : "Normal", "Wind" : "Strong", "label" : "No"}
	data[6] = {"Outlook" : "Overcast", "Temp" : "Cool", "Humidity" : "Normal", "Wind" : "Strong", "label" : "Yes"}
	data[7] = {"Outlook" : "Sunny", "Temp" : "Mild", "Humidity" : "High", "Wind" : "Weak", "label" : "No"}
	data[8] = {"Outlook" : "Sunny", "Temp" : "Cool", "Humidity" : "Normal", "Wind" : "Weak", "label" : "Yes"}
	data[9] = {"Outlook" : "Rain", "Temp" : "Mild", "Humidity" : "Normal", "Wind" : "Weak", "label" : "Yes"}
	data[10] = {"Outlook" : "Sunny", "Temp" : "Mild", "Humidity" : "Normal", "Wind" : "Strong", "label" : "Yes"}
	data[11] = {"Outlook" : "Overcast", "Temp" : "Mild", "Humidity" : "High", "Wind" : "Strong", "label" : "Yes"}
	data[12] = {"Outlook" : "Overcast", "Temp" : "Hot", "Humidity" : "Normal", "Wind" : "Weak", "label" : "Yes"}
	data[13] = {"Outlook" : "Rain", "Temp" : "Mild", "Humidity" : "High", "Wind" : "Strong", "label" : "No"}

	return data, attributes

"""
Split data into training set and test set
"""
def split(data):
	example = list(data)
	test = []
	number_of_test = math.trunc(len(data) * 0.2)
	for i in range(number_of_test):
		random_number = randint(0, len(example) - 1)
		test.append(example[random_number])
		del example[random_number]

	return example, test

"""
calculate entropy of the giving set
"""
def entropy(examples):
	labels = {}
	for example in examples:
		if example["label"] in labels:
			labels[example["label"]] = labels[example["label"]] + 1
		else:
			labels[example["label"]] = 1

	num = 0
	for label in labels:
		p = float(labels[label]) / len(examples)
		num = num - p * math.log(p, 2)

	return num

"""
find the best attribute using info gain
"""
def find_best_attribute(examples, attributes):
	best_attribute = None
	max_info_gain = 0
	entropy_s = entropy(examples)
	for attribute in attributes:
		branch = {}
		info_gain = entropy_s
		for example in examples:
			if example[attribute] not in branch:
				branch[example[attribute]] = [example]
			else:
				branch[example[attribute]].append(example)

		for value_of_attribute in attributes[attribute]:
			if value_of_attribute in branch:
				info_gain = info_gain - entropy(branch[value_of_attribute]) * len(branch[value_of_attribute]) / len(examples)

		if info_gain > max_info_gain:
			max_info_gain = info_gain
			best_attribute = attribute

	return best_attribute

"""
dfs, result_from_decision_tree and generate_error_rate are used for generating error rate
"""
def dfs(node, instance):
	if node.is_leaf == True:
		return node.attribute
	else:
		return dfs(node.children[instance[node.attribute]], instance)

def result_from_decision_tree(instance):
	return dfs(root, instance)

def generate_error_rate():
	global origin_examples
	global tests
	train_error_count = 0
	for exm in origin_examples:
		if result_from_decision_tree(exm) != exm["label"]:
			train_error_count = train_error_count + 1

	test_error_count = 0
	for test in tests:
		if result_from_decision_tree(test) != test["label"]:
			test_error_count = test_error_count + 1

	training_error.append(float(train_error_count) / len(origin_examples))
	test_error.append(float(test_error_count) / len(tests))

	return 0

"""
main working function
"""
def id3(examples, attributes):
	# create a a tree node
	current_node = Tree()
	global root
	if root == None:
		root = current_node

	set_of_label = set()
	list_of_label = []
	for example in examples:
		set_of_label.add(example["label"])
		list_of_label.append(example["label"])

	# if examples only are all have the same label, assign the label to the tree node
	if len(set_of_label) == 1:
		current_node.attribute = set_of_label.pop()
		return current_node

	max_num = 0
	max_label = None
	for current_label in set_of_label:
		current_num = list_of_label.count(current_label)
		if current_num > max_num:
			max_num = current_num
			max_label = current_label

	# if examples does not have any more attributes, assign the major label to the tree node
	if len(attributes) == 0:
		current_node.attribute = max_label
		return current_node

	# find the best attribute to make partition
	best_attribute_name = find_best_attribute(examples, attributes)
	current_node.attribute = best_attribute_name
	current_node.is_leaf = False

	# partition examples into different branch
	branch = {}
	for example in examples:
		if example[best_attribute_name] not in branch:
			branch[example[best_attribute_name]] = [example]
		else:
			branch[example[best_attribute_name]].append(example)

	# generate error rates
	for value_of_attribute in attributes[best_attribute_name]:
		if value_of_attribute in branch:
			max_branch_label = None
			max_num = 0
			for current_label in set_of_label:
				current_num = branch[value_of_attribute].count(current_label)
				if current_num > max_num:
					max_num = current_num
					max_branch_label = current_label
			child = Tree()
			child.attribute = max_branch_label
		else:
			child = Tree()
			child.attribute = max_label

		current_node.children[value_of_attribute] = child
	generate_error_rate()
	
	#iteratively generate more nodes
	for value_of_attribute in attributes[best_attribute_name]:
		if value_of_attribute in branch:
			new_attributes = dict(attributes)
			del new_attributes[best_attribute_name]
			child = id3(branch[value_of_attribute], new_attributes)
			current_node.children[value_of_attribute] = child

	return current_node

def main(argv):
	global origin_examples
	global tests
	#play tennis example

	data, attributes = import_data()
	examples, tests = split(data)
	origin_examples = list(examples)
	id3(examples, attributes)
	generate_error_rate()

	# output error rate
	print ('[%s]' % ', '.join(map(str, training_error)))
	print ('[%s]' % ', '.join(map(str, test_error)))

	#HR data people leaving the company in three years
	data, attributes = HRData()
	examples, tests = split(data)
	origin_examples = list(examples)
	id3(examples, attributes)
	generate_error_rate()

	# output error rate
	print ('[%s]' % ', '.join(map(str, training_error)))
	print ('[%s]' % ', '.join(map(str, test_error)))
	






if __name__ == "__main__":
    main(sys.argv)