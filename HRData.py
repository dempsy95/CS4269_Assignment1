import csv
import random


def HRData():
	features=["promotion_in_5_years","work_accident", "married","worked_for_more_than_5_companies",
	"work_in_tech", "high_satisfaction", "high_evaluation", "high_salary", "average_weekly_hours_exceed_50",
	"work_in_sales"]
	attributes={}
	values={"Yes":1, "No":-1}
	for feature in features: 
		attributes[feature]={'Yes', 'No'}

	instances=[]
	for i in range(100):
		instance={}
		for feature in features: 
			instance[feature]=random.sample(['Yes', 'No'],  1)[0]
		instances.append(instance)
		
		label=-values[instance["promotion_in_5_years"]]*0.3+values[instance["work_accident"]]*0.5\
		-values[instance["high_salary"]]*0.45-values[instance["high_evaluation"]]*0.35\
		+values[instance["average_weekly_hours_exceed_50"]]*0.5\
		-values[instance["high_satisfaction"]]*0.4\
		+values[instance["worked_for_more_than_5_companies"]]*0.6
		if abs(label-1)> abs(label+1): 
			instance['label']="Yes"
		else: 
			instance['lable']="No"

	return attributes,instances


attributes,instances=HRData()
print(attributes)